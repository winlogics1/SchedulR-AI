import os
from plyer import notification
import winsound  # Windows built-in sound
import cv2
import json
import sqlite3
import re
import io
import wave
import struct
import math
import base64
import imghdr
import time
import pytesseract
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime, timedelta
from dateutil import parser
from openai import OpenAI
from apscheduler.schedulers.background import BackgroundScheduler

# ==========================================
# 0. System & App Configuration
# ==========================================
st.set_page_config(
    page_title="AI Meeting Scheduler",
    page_icon="📅",
    layout="wide"
)

# Configure Tesseract execution path
if os.path.exists('/usr/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
elif os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize OpenAI API Client
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY", "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"),
    base_url="https://openrouter.ai/api/v1"
)

# Initialize background scheduler in Streamlit session state
if "scheduler" not in st.session_state:
    scheduler = BackgroundScheduler()
    scheduler.start()
    st.session_state.scheduler = scheduler

if "alarms" not in st.session_state:
    st.session_state.alarms = []

if "alarm_active" not in st.session_state:
    st.session_state.alarm_active = False

if "pending_conflict_meeting" not in st.session_state:
    st.session_state.pending_conflict_meeting = None

# ==========================================
# 1. Helper Functions (Audio & DB)
# ==========================================
def generate_beep_wav(frequency=1000.0, duration=1.5, sample_rate=44100) -> bytes:
    """Generates a WAV audio file in memory without external audio dependencies."""
    num_samples = int(duration * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        for i in range(num_samples):
            t = float(i) / sample_rate
            value = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * frequency * t))
            wav_file.writeframes(struct.pack('<h', value))
    return buf.getvalue()

def init_db():
    """Initializes SQLite database to store scheduled meetings."""
    conn = sqlite3.connect("meetings.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            meeting_time TEXT,
            reminder_time TEXT,
            venue TEXT,
            status TEXT DEFAULT 'PENDING'
        )
    """)
    conn.commit()
    conn.close()

def find_conflicting_meetings(meeting_dt: datetime) -> list:
    """Finds existing meetings scheduled at the exact same date and time."""
    conn = sqlite3.connect("meetings.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title, meeting_time, venue FROM meetings WHERE meeting_time = ?",
        (meeting_dt.isoformat(),)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows

def store_meeting(title: str, meeting_dt: datetime, reminder_dt: datetime, venue: str):
    """Saves meeting record into the database."""
    conn = sqlite3.connect("meetings.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO meetings (title, meeting_time, reminder_time, venue)
        VALUES (?, ?, ?, ?)
    """, (title, meeting_dt.isoformat(), reminder_dt.isoformat(), venue))
    conn.commit()
    conn.close()

def get_all_meetings():
    """Fetches stored meetings from SQLite."""
    conn = sqlite3.connect("meetings.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, meeting_time, reminder_time, venue, status FROM meetings ORDER BY meeting_time ASC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def delete_meeting(meeting_id: int):
    """Deletes a meeting from SQLite database."""
    conn = sqlite3.connect("meetings.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
    conn.commit()
    conn.close()

# ==========================================
# 2. Image Preprocessing & OCR
# ==========================================
def image_bytes_to_data_url(image_bytes: bytes) -> str:
    image_type = imghdr.what(None, h=image_bytes) or "jpeg"
    base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{image_type};base64,{base64_encoded}"

def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")

    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def perform_ocr(image_bytes: bytes) -> str:
    thresh_img = preprocess_image_bytes(image_bytes)
    pil_img = Image.fromarray(thresh_img)
    return pytesseract.image_to_string(pil_img)

# ==========================================
# 3. Parsing Engine
# ==========================================
def fallback_local_parse(raw_text: str) -> dict:
    title_match = re.search(r'What:\s*(.*)', raw_text)
    title = title_match.group(1).strip() if title_match else None

    meet_match = re.search(r'https?://meet\.google\.com/[a-z0-9-]+', raw_text)
    venue = meet_match.group(0) if meet_match else "Not Specified"

    date_match = re.search(
        r'([A-Za-z]+,\s+[A-Za-z]+\s+\d{1,2})\s*[\+\•\·\-]?\s*(\d{1,2}:\d{2})\s*(?:[\-\–]\s*\d{1,2}:\d{2}\s*)?([ap][mne\.]*)?',
        raw_text,
        re.IGNORECASE
    )

    if date_match:
        date_part = date_match.group(1)
        time_part = date_match.group(2)
        meridiem_part = date_match.group(3) or ""

        if re.search(r'p[mne\.]*', meridiem_part, re.IGNORECASE) or re.search(r'\d{1,2}:\d{2}\s*p', raw_text, re.IGNORECASE):
            ampm = "PM"
        elif re.search(r'a[mne\.]*', meridiem_part, re.IGNORECASE):
            ampm = "AM"
        else:
            ampm = "PM" if "pm" in raw_text.lower() else "AM"

        full_dt_str = f"{date_part} {datetime.now().year} {time_part} {ampm}"
        try:
            parsed_dt = parser.parse(full_dt_str)
            meeting_datetime = parsed_dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            meeting_datetime = None
    else:
        meeting_datetime = None

    return {
        "title": title if title else "Upcoming Meeting",
        "meeting_datetime": meeting_datetime,
        "venue": venue
    }

def extract_meeting_details_via_openai_vision(image_bytes: bytes, max_retries: int = 3) -> dict:
    now = datetime.now()
    current_date_str = now.strftime("%Y-%m-%d")
    current_time_str = now.strftime("%H:%M:%S")
    current_day_str = now.strftime("%A")
    tomorrow_date_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    data_url = image_bytes_to_data_url(image_bytes)

    prompt = f"""
    You are an expert meeting scheduler assistant. Analyze this image (handwriting or text screenshot) and extract meeting details directly into JSON format.

    CURRENT SYSTEM CONTEXT:
    - Today's Exact Date: {current_date_str} ({current_day_str})
    - Today's Exact Time: {current_time_str}
    - Tomorrow's Date: {tomorrow_date_str}

    RELATIVE DATE RULES (Including Roman Urdu / Hindi / English):
    - "aj", "aaj", "today" -> Use date {current_date_str}
    - "kal", "tomorrow" -> Use date {tomorrow_date_str}
    - If specific day name is mentioned (e.g. "Monday", "Friday"), resolve relative to today ({current_day_str}).
    - If NO specific time is mentioned, default to standard business time like "09:00:00" or near current time.

    Output Fields:
    1. "title": The main meeting title/subject.
    2. "meeting_datetime": Start date & time formatted strictly as ISO 8601: "YYYY-MM-DD HH:MM:SS".
    3. "venue": Meeting location or virtual link. If not found, use "Not Specified".

    If NO meeting details can be identified, set "meeting_datetime" to null.

    Respond ONLY with a valid raw JSON object:
    {{"title": "...", "meeting_datetime": "YYYY-MM-DD HH:MM:SS" or null, "venue": "..."}}
    """

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            result_text = response.choices[0].message.content.strip()

            if "<html" in result_text.lower() or "502 bad gateway" in result_text.lower():
                time.sleep(1.5 * (attempt + 1))
                continue

            clean_json_str = result_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json_str)

        except Exception as e:
            error_msg = str(e)
            if "502" in error_msg or "<html" in error_msg.lower():
                if attempt < max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                return {"error": "Server temporarily unavailable. Please retry in a few seconds."}
            return {"error": f"Failed to parse handwriting via GPT-4o-mini vision: {error_msg}"}

    return {"error": "Service unavailable after multiple attempts. Please retry."}

def extract_meeting_details_with_openai(raw_text: str) -> dict:
    now = datetime.now()
    current_date_str = now.strftime("%Y-%m-%d")
    current_time_str = now.strftime("%H:%M:%S")
    current_day_str = now.strftime("%A")
    tomorrow_date_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    prompt = f"""
    You are an expert meeting scheduler assistant. Extract meeting details from this OCR or input text:

    ---
    {raw_text}
    ---

    CURRENT SYSTEM CONTEXT:
    - Today's Exact Date: {current_date_str} ({current_day_str})
    - Today's Exact Time: {current_time_str}
    - Tomorrow's Date: {tomorrow_date_str}

    RELATIVE DATE RULES (Including Roman Urdu / Hindi / English):
    - "aj", "aaj", "today" -> Use date {current_date_str}
    - "kal", "tomorrow" -> Use date {tomorrow_date_str}
    - "parson", "day after tomorrow" -> Use date {(now + timedelta(days=2)).strftime("%Y-%m-%d")}
    - If specific day name is mentioned (e.g. "Monday", "Friday"), calculate the exact target date starting from today ({current_day_str}).
    - If time is missing or incomplete, assume current time ({current_time_str}) or next rounded hour.

    Output Requirements:
    1. "title": The main meeting title/subject (e.g., "Python and Java Meeting").
    2. "meeting_datetime": The EXACT start date and time formatted in ISO 8601: "YYYY-MM-DD HH:MM:SS".
    3. "venue": Location or meeting link. If not found, use "Not Specified".

    If NO meeting date/time can be identified, set "meeting_datetime" to null.

    Respond ONLY with a valid raw JSON object.
    Target format:
    {{"title": "...", "meeting_datetime": "YYYY-MM-DD HH:MM:SS" or null, "venue": "..."}}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise JSON extraction engine with strong temporal reasoning."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        result_text = response.choices[0].message.content.strip()

        if "<html" in result_text.lower() or "502 bad gateway" in result_text.lower():
            return fallback_local_parse(raw_text)

        clean_json_str = result_text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json_str)

    except Exception:
        return fallback_local_parse(raw_text)

# ==========================================
# 4. Alarm Callbacks & Scheduling
# ==========================================
def trigger_alarm_callback(title: str, meeting_time_str: str, venue: str, lead_min: int = 60):
    st.session_state.alarm_active = True
    st.session_state.alarms.append({
        "title": title,
        "meeting_time": meeting_time_str,
        "venue": venue,
        "lead_min": lead_min,
        "triggered_at": datetime.now().strftime("%I:%M:%S %p")
    })

    try:
        notification.notify(
            title=f"⏰ Meeting Reminder: {title}",
            message=f"Starts in {lead_min} min ({meeting_time_str}) | Venue: {venue}",
            app_name="AI Scheduler",
            timeout=10
        )
    except Exception:
        pass

    try:
        winsound.Beep(1000, 1500)
    except Exception:
        pass

def schedule_alarms_for_meeting(title: str, meeting_dt: datetime, venue: str, offsets_minutes: list):
    now = datetime.now()
    time_until_meeting = (meeting_dt - now).total_seconds() / 60.0

    # If meeting is within 2 minutes (or seconds away), trigger IMMEDIATELY
    if time_until_meeting <= 2.0:
        trigger_alarm_callback(
            title=title,
            meeting_time_str=meeting_dt.strftime('%I:%M %p'),
            venue=venue,
            lead_min=max(0, int(round(time_until_meeting)))
        )
        st.warning(f"🚨 **IMMEDIATE ALARM:** Meeting starting in {round(max(0, time_until_meeting), 1)} minute(s)!")
        return 1

    scheduled_count = 0
    for offset in offsets_minutes:
        reminder_dt = meeting_dt - timedelta(minutes=offset)
        
        if reminder_dt <= now:
            trigger_alarm_callback(title, meeting_dt.strftime('%I:%M %p'), venue, lead_min=offset)
            scheduled_count += 1
        else:
            st.session_state.scheduler.add_job(
                trigger_alarm_callback,
                'date',
                run_date=reminder_dt,
                args=[title, meeting_dt.strftime('%I:%M %p'), venue, offset]
            )
            scheduled_count += 1
            st.info(f"⏰ Alarm set for {offset} minute(s) before meeting ({reminder_dt.strftime('%Y-%m-%d %I:%M %p')})")

    return scheduled_count

# ==========================================
# 5. Streamlit User Interface
# ==========================================
def main():
    init_db()

    st.title("📅 AI Automated Meeting Scheduler")
    st.caption("Upload meeting screenshots to extract events, detect time conflicts, and configure multi-alarm notifications.")

    # Sidebar Options for User-defined Alarms
    st.sidebar.header("⏰ Alarm Settings")
    
    num_alarms = st.sidebar.number_input(
        "Number of Alarms per Meeting", 
        min_value=1, 
        max_value=5, 
        value=1, 
        step=1,
        help="Select between 1 to 5 alarms per event."
    )

    alarm_offsets = []
    st.sidebar.markdown("**Set Lead Times (Minutes):**")
    for i in range(int(num_alarms)):
        offset = st.sidebar.slider(
            f"Alarm #{i+1} (Minutes before event)", 
            min_value=1, 
            max_value=60, 
            value=min(15 * (i + 1), 60), 
            key=f"alarm_slider_{i}"
        )
        alarm_offsets.append(offset)

    # Display Active Triggered Alarms
    if st.session_state.alarms or st.session_state.alarm_active:
        st.session_state.alarm_active = True
        st.error("⏰ **MEETING ALARM TRIGGERED!**")
        
        if st.session_state.alarms:
            for active_alarm in st.session_state.alarms:
                st.markdown(f"**Title:** {active_alarm['title']}  \n"
                            f"**Time:** Starts in {active_alarm.get('lead_min', 0)} mins ({active_alarm['meeting_time']})  \n"
                            f"**Venue:** [{active_alarm['venue']}]({active_alarm['venue']})")
                st.divider()
        
        st.audio(generate_beep_wav(), format="audio/wav", autoplay=True, loop=True)

        if st.button("🔕 Stop Alarm", type="primary"):
            st.session_state.alarm_active = False
            st.session_state.alarms.clear()
            st.rerun()

        st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Meeting Screenshot")
        
        image_type_option = st.segmented_control(
            "Select Image Type / Extraction Path:",
            options=["Screenshot (typed text)", "Handwritten note", "Auto-detect"],
            default="Auto-detect"
        )
        
        uploader_key = st.session_state.get("uploader_key", 0)
        uploaded_file = st.file_uploader(
            "Choose an image (PNG, JPG, JPEG)", 
            type=["png", "jpg", "jpeg"],
            key=f"uploader_{uploader_key}"
        )

        # ----------------------------------------------------
        # Conflict Confirmation Prompt
        # ----------------------------------------------------
        if st.session_state.pending_conflict_meeting:
            pending = st.session_state.pending_conflict_meeting
            conflicts = pending["conflicts"]
            
            st.warning("⚠️ **SCHEDULING CONFLICT DETECTED!**")
            st.write(f"You already have **{len(conflicts)} meeting(s)** scheduled at **{pending['meeting_dt'].strftime('%Y-%m-%d %I:%M %p')}**:")
            
            for c in conflicts:
                st.markdown(f"- 📌 **{c[1]}** (Venue: {c[3]})")
                
            st.info("Do you still want to schedule this new meeting alongside your existing one?")
            
            c_btn1, c_btn2 = st.columns(2)
            
            with c_btn1:
                if st.button("✅ Yes, Add Meeting Anyway", type="primary"):
                    store_meeting(pending["title"], pending["meeting_dt"], pending["reminder_dt"], pending["venue"])
                    schedule_alarms_for_meeting(pending["title"], pending["meeting_dt"], pending["venue"], pending["offsets"])
                    
                    st.success("✅ Overlapping Meeting Added Successfully!")
                    st.session_state.pending_conflict_meeting = None
                    st.rerun()

            with c_btn2:
                if st.button("❌ No, Cancel & Discard Picture", type="secondary"):
                    st.session_state.pending_conflict_meeting = None
                    st.session_state.uploader_key = uploader_key + 1
                    st.info("🗑️ Process cancelled. Image removed.")
                    st.rerun()

        # ----------------------------------------------------
        # Standard Upload & Extraction Flow
        # ----------------------------------------------------
        elif uploaded_file is not None:
            image_bytes = uploaded_file.read()
            st.image(image_bytes, caption="Uploaded Image", use_container_width=True)

            if st.button("⚡ Process Image & Schedule", type="primary"):
                with st.spinner("Extracting & Processing Details via GPT-4o-mini..."):
                    parsed_data = {}
                    
                    try:
                        if image_type_option == "Handwritten note":
                            parsed_data = extract_meeting_details_via_openai_vision(image_bytes)
                        elif image_type_option == "Screenshot (typed text)":
                            raw_text = perform_ocr(image_bytes)
                            parsed_data = extract_meeting_details_with_openai(raw_text)
                        else:  # Auto-detect
                            try:
                                raw_text = perform_ocr(image_bytes)
                                clean_text = raw_text.strip()
                                alnum_count = sum(1 for char in clean_text if char.isalnum())
                                
                                if len(clean_text) >= 10 and alnum_count >= 5:
                                    parsed_data = extract_meeting_details_with_openai(raw_text)
                                else:
                                    parsed_data = extract_meeting_details_via_openai_vision(image_bytes)
                            except Exception:
                                parsed_data = extract_meeting_details_via_openai_vision(image_bytes)

                    except Exception as e:
                        st.error(f"⚠️ Error processing image: {str(e)}")
                        return

                    if "error" in parsed_data:
                        st.error(f"❌ {parsed_data['error']}")
                        st.info("💡 Please try uploading a clearer image or retry processing.")
                        return

                    meeting_datetime_str = parsed_data.get("meeting_datetime")
                    if not meeting_datetime_str:
                        st.error("❌ NO MEETING DETAILS FOUND IN THE IMAGE!")
                        return

                    try:
                        title = parsed_data.get("title") or "Upcoming Meeting"
                        venue = parsed_data.get("venue") or "Not Specified"
                        meeting_dt = datetime.strptime(meeting_datetime_str, "%Y-%m-%d %H:%M:%S")
                        reminder_dt = meeting_dt - timedelta(minutes=alarm_offsets[0])
                    except Exception as e:
                        st.error(f"❌ Error parsing date details: {str(e)}")
                        return

                    now = datetime.now()
                    
                    # 1-minute past tolerance window fix
                    if meeting_dt < (now - timedelta(seconds=60)):
                        st.warning("⚠️ The meeting date has already passed!")
                        return

                    # Check for date/time conflicts in database
                    conflicts = find_conflicting_meetings(meeting_dt)
                    if conflicts:
                        st.session_state.pending_conflict_meeting = {
                            "title": title,
                            "meeting_dt": meeting_dt,
                            "reminder_dt": reminder_dt,
                            "venue": venue,
                            "offsets": alarm_offsets,
                            "conflicts": conflicts
                        }
                        st.rerun()

                    # Save and Schedule directly if no conflict exists
                    store_meeting(title, meeting_dt, reminder_dt, venue)
                    st.success("✅ Meeting Successfully Scheduled!")
                    st.json({
                        "Title": title,
                        "Meeting Datetime": meeting_dt.strftime('%Y-%m-%d %I:%M %p'),
                        "Configured Alarms Count": len(alarm_offsets),
                        "Venue": venue
                    })

                    schedule_alarms_for_meeting(title, meeting_dt, venue, alarm_offsets)

    with col2:
        st.subheader("📋 Scheduled Meetings (Database)")
        meetings = get_all_meetings()

        if not meetings:
            st.info("No scheduled meetings found in database.")
        else:
            for m in meetings:
                m_id, m_title, m_time, r_time, m_venue, m_status = m
                with st.expander(f"📌 {m_title}", expanded=True):
                    st.markdown(f"**Meeting Time:** {m_time}")
                    st.markdown(f"**Reminder Time:** {r_time}")
                    st.markdown(f"**Venue:** [{m_venue}]({m_venue})" if "http" in m_venue else f"**Venue:** {m_venue}")
                    
                    if st.button("🗑️ Delete Event", key=f"del_{m_id}"):
                        delete_meeting(m_id)
                        st.rerun()

if __name__ == "__main__":
    main()