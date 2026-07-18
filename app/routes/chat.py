from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict
from groq import Groq
from supabase import create_client
from app.config import SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY
from app.auth import verify_token
import uuid
import re
import time
import logging
from datetime import date, timedelta, datetime

# ── Structured logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | chat | %(message)s",
)
logger = logging.getLogger("fixpro_chat")

router = APIRouter()
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Rate limiting ────────────────────────────────────────────────────────────────
# Simple sliding-window limiter keyed by session_id, in-process.
# Good enough for a single-instance Railway deployment.
RATE_LIMIT_MAX_MESSAGES = 15
RATE_LIMIT_WINDOW_SECONDS = 60
_rate_limit_log: Dict[str, list] = {}

def is_rate_limited(session_id: str) -> bool:
    now = time.time()
    timestamps = _rate_limit_log.get(session_id, [])
    timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SECONDS]
    timestamps.append(now)
    _rate_limit_log[session_id] = timestamps
    if len(timestamps) > RATE_LIMIT_MAX_MESSAGES:
        logger.warning(f"Rate limit hit for session {session_id} ({len(timestamps)} msgs/{RATE_LIMIT_WINDOW_SECONDS}s)")
        return True
    return False

# ── Safe Groq wrapper (graceful fallback on outage/timeout) ──────────────────────
def safe_groq_call(messages: list, max_tokens: int = 300, temperature: float = 0.4, fallback: str = None) -> str:
    try:
        res = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=12,
        )
        return res.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        return fallback or "Sorry, I'm having trouble responding right now. Please call us at +92 300 1234567 or try again in a moment."

# ── Session cache is defined later near the persistent session functions ──────

# ══════════════════════════════════════════════════════════════════════════════
# RAG — Shop Knowledge Base (FixPro iPhone Repair, Lahore)
# ══════════════════════════════════════════════════════════════════════════════
SHOP_RAG = """
=== FixPro iPhone Repair — Shop Information ===

SHOP NAME: FixPro iPhone Repair
LOCATION: Shop 14, Al-Hamra Arcade, Main Boulevard Gulberg III, Lahore
PHONE: +92 300 1234567
GOOGLE MAPS: https://maps.google.com/?q=FixPro+Gulberg+Lahore
HOURS: Monday–Saturday: 10:00 AM – 8:00 PM | Sunday: Closed

SERVICES & PRICES:
- Screen Repair (Original):     Rs 8,000 – 14,000  | Time: 1–2 hours
- Screen Repair (Compatible):   Rs 4,000 – 7,000   | Time: 1–2 hours
- Battery Replacement:          Rs 3,500 – 5,500   | Time: 30–45 minutes
- Charging Port Repair:         Rs 2,500 – 4,000   | Time: 1–2 hours
- Camera Repair (Rear):         Rs 5,000 – 9,000   | Time: 2–3 hours
- Camera Repair (Front):        Rs 3,000 – 5,000   | Time: 1–2 hours
- Water Damage Repair:          Rs 4,000 – 10,000  | Time: 24–48 hours (assessment first)
- Software Issues / Reset:      Rs 1,500 – 3,000   | Time: 1–3 hours
- Back Glass Repair:            Rs 3,500 – 6,000   | Time: 2–3 hours
- Speaker / Mic Repair:         Rs 2,000 – 4,000   | Time: 1–2 hours

SUPPORTED MODELS: iPhone 8 and above (iPhone 8, X, XS, XR, 11, 12, 13, 14, 15 series, all Pro/Max variants)

WARRANTY:
- All repairs come with a 30-day warranty on parts and labor
- Water damage repairs: 7-day warranty (due to nature of damage)
- Warranty covers same issue only — does not cover new physical damage

PAYMENT METHODS: Cash, EasyPaisa, JazzCash, Bank Transfer

FAQS:
Q: Do I need an appointment?
A: Walk-ins welcome but appointments are recommended to avoid waiting.

Q: How long does repair take?
A: Most repairs done same day. Screen/battery: 30 min–2 hrs. Water damage: 24–48 hrs.

Q: Do you use original parts?
A: We offer both original (OEM) and high-quality compatible parts. We recommend original for best quality.

Q: Is my data safe during repair?
A: Yes. We never access your data. You can set a passcode before handing in.

Q: What if my phone can't be repaired?
A: We only charge if we successfully fix it. No fix = no fee (except diagnostic fee of Rs 500).

Q: Can I get a price quote?
A: Yes! Prices depend on iPhone model. Share your model and issue for an exact quote.
"""

WAIT_TIMES = {
    "screen": "1–2 hours",
    "battery": "30–45 minutes",
    "charging": "1–2 hours",
    "port": "1–2 hours",
    "camera": "2–3 hours",
    "water": "24–48 hours",
    "software": "1–3 hours",
    "back glass": "2–3 hours",
    "speaker": "1–2 hours",
    "mic": "1–2 hours",
}

def get_wait_time(issue: str) -> str:
    issue_lower = issue.lower()
    for key, time in WAIT_TIMES.items():
        if key in issue_lower:
            return time
    return "1–3 hours"

# ── Models ─────────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class OwnerChatRequest(BaseModel):
    messages: List[Message]
    context: Optional[dict] = None

class CustomerChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: Optional[List[Message]] = []

# ── Response builder (always consistent shape) ─────────────────────────────────
def respond(
    reply: str,
    session_id: str,
    booking_created: bool = False,
    booking_info: dict = None,
    session_reset: bool = False,
    quick_replies: list = None,
    slot_buttons: list = None,
    time_buttons: list = None,
    rating_prompt: bool = False,
    typing_delay_ms: int = 600,
):
    return {
        "reply": reply,
        "session_id": session_id,
        "booking_created": booking_created,
        "booking_info": booking_info,
        "session_reset": session_reset,
        "quick_replies": quick_replies or [],
        "slot_buttons": slot_buttons or [],
        "time_buttons": time_buttons or [],
        "rating_prompt": rating_prompt,
        "typing_delay_ms": typing_delay_ms,
    }

# ── Profanity / spam filter ────────────────────────────────────────────────────
SPAM_PATTERNS = [
    r'\b(fuck|shit|bastard|asshole|bitch|harami|gandu|madarchod|benchod)\b',
]
def is_spam(text: str) -> bool:
    t = text.strip().lower()
    for p in SPAM_PATTERNS:
        if re.search(p, t): return True
    if len(text) > 500: return True
    # Never flag plausible dates/times/phone numbers as symbol-spam
    if re.match(r'^\d{4}-\d{2}-\d{2}$', t): return False
    if re.match(r'^\d{1,2}:\d{2}\s*(am|pm)?$', t): return False
    if re.match(r'^[\d\s+\-()]{7,}$', t): return False
    if re.match(r'^[^a-zA-Z\u0600-\u06FF\s]{10,}$', text): return True
    return False

# ── Exit intent ────────────────────────────────────────────────────────────────
EXIT_WORDS = ['cancel', 'never mind', 'nevermind', 'bye', 'goodbye', 'chor do',
              'rehne do', 'nahi chahiye', 'band karo', 'chordo', 'choro']
def is_exit(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in EXIT_WORDS)

# ── Reschedule intent ──────────────────────────────────────────────────────────
RESCHEDULE_WORDS = ['reschedule', 'change date', 'change time', 'date change', 'time change',
                    'date badlo', 'waqt badlo', 'reschedule karna', 'postpone']
def is_reschedule(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in RESCHEDULE_WORDS)

# ── Cancel booking intent ──────────────────────────────────────────────────────
CANCEL_WORDS = ['cancel booking', 'cancel my booking', 'cancel appointment',
                'booking cancel', 'appointment cancel', 'booking cancel karna',
                'meri booking cancel', 'cancel karo']
def is_cancel_booking(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in CANCEL_WORDS)

# ── Language detection ─────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    urdu_chars = re.findall(r'[\u0600-\u06FF]', text)
    if len(urdu_chars) > 2: return 'urdu'
    roman_urdu = ['kya', 'hai', 'mera', 'meri', 'aap', 'hum', 'yeh', 'woh', 'nahi',
                  'haan', 'theek', 'kal', 'aj', 'aaj', 'phone', 'naam', 'karo',
                  'bhai', 'yaar', 'ap', 'ho', 'kar', 'karo', 'chahiye', 'batao']
    words = text.lower().split()
    if sum(1 for w in words if w in roman_urdu) >= 2: return 'roman_urdu'
    return 'english'

import re as _re
from difflib import get_close_matches

# ── Smart date parser (with typo tolerance) ────────────────────────────────────
def parse_date(text: str) -> Optional[str]:
    text = text.lower().strip()
    today = date.today()
    if any(w in text for w in ['today', 'aaj', 'aj']): return str(today)
    if any(w in text for w in ['tomorrow', 'tommorow', 'tomorow', 'tomarrow', 'kal', 'kl']): return str(today + timedelta(days=1))
    if any(w in text for w in ['day after tomorrow', 'parso', 'parsoon']): return str(today + timedelta(days=2))
    days = {
        'monday': 0, 'somwar': 0,
        'tuesday': 1, 'mangal': 1,
        'wednesday': 2, 'budh': 2,
        'thursday': 3, 'jumeraat': 3, 'jumerat': 3,
        'friday': 4, 'jumma': 4, 'juma': 4,
        'saturday': 5, 'hafta': 5,
        'sunday': 6, 'itwar': 6,
    }
    # Exact substring match first
    for day_name, day_num in days.items():
        if day_name in text:
            days_ahead = day_num - today.weekday()
            if days_ahead <= 0: days_ahead += 7
            return str(today + timedelta(days=days_ahead))
    # Fuzzy match each word against day names to catch typos (fruday, satrday, etc.)
    words = re.findall(r'[a-z]+', text)
    for word in words:
        if len(word) < 4: continue
        match = get_close_matches(word, days.keys(), n=1, cutoff=0.72)
        if match:
            day_num = days[match[0]]
            days_ahead = day_num - today.weekday()
            if days_ahead <= 0: days_ahead += 7
            return str(today + timedelta(days=days_ahead))
    patterns = [r'(\d{4}-\d{2}-\d{2})']
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                d = datetime.strptime(m.group(0), '%Y-%m-%d').date()
                if d >= today: return str(d)
            except: pass
    return None

# ── Fuzzy yes/no/skip detection (typo tolerant) ────────────────────────────────
YES_WORDS = ['yes', 'y', 'haan', 'ha', 'confirm', 'ok', 'okay', 'theek hai', 'theek', 'ji', 'yes ✅', 'sure', 'yep', 'yeah']
NO_WORDS = ['no', 'n', 'nahi', 'cancel', 'nope', 'no ❌', 'nah']
SKIP_WORDS = ['skip', 'no', 'nahi', 'nope', "don't have", 'dont have', 'na', 'none']

def fuzzy_match_word(msg: str, word_list: list, cutoff: float = 0.75) -> bool:
    t = msg.lower().strip()
    if t in word_list: return True
    for w in word_list:
        if len(w) < 3: continue
        if get_close_matches(t, [w], n=1, cutoff=cutoff): return True
    return False

def is_yes(msg: str) -> bool: return fuzzy_match_word(msg, YES_WORDS)
def is_no(msg: str) -> bool: return fuzzy_match_word(msg, NO_WORDS)
def is_skip(msg: str) -> bool: return fuzzy_match_word(msg, SKIP_WORDS)

# ── Detect off-track questions during data collection ──────────────────────────
QUESTION_TRIGGERS = ['?', 'how much', 'price', 'cost', 'hours', 'open', 'close',
                     'location', 'where', 'address', 'warranty', 'kitna', 'kahan',
                     'kab', 'kya hai', 'timing', 'kitne']

def looks_like_question(msg: str, expected_step: str) -> bool:
    """Returns True if the message looks like an off-topic question rather than
    a plausible answer to the current collection step."""
    t = msg.lower().strip()
    # Plausible answers to specific steps should never be treated as questions
    if expected_step == "get_phone" and re.search(r'\d{7,}', t): return False
    if expected_step == "get_email" and ('@' in t or is_skip(t)): return False
    if expected_step == "get_name" and len(t.split()) <= 4 and not any(q in t for q in QUESTION_TRIGGERS): return False
    if expected_step in ("get_date", "get_new_date") and (parse_date(t) or any(d in t for d in
        ['monday','tuesday','wednesday','thursday','friday','saturday','sunday','tomorrow','today'])): return False
    if expected_step in ("get_time", "get_new_time") and re.search(r'\d{1,2}(:\d{2})?\s*(am|pm)?', t): return False
    return any(q in t for q in QUESTION_TRIGGERS)

def answer_offtrack_question(msg: str, lang: str, history: list) -> str:
    """Quick RAG answer for a question asked mid-flow, kept short."""
    system_prompt = f"""You are a helpful assistant for FixPro iPhone Repair in Lahore.
Answer the customer's question briefly using the shop info below (max 60 words).
Reply in the same language as the customer (English, Roman Urdu, or Urdu).
Do NOT ask if they want to book — they're already mid-booking, just answer and we'll re-ask the booking question after.

{SHOP_RAG}"""
    try:
        res = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": msg}],
            max_tokens=150, temperature=0.4,
            timeout=10,
        )
        return res.choices[0].message.content
    except Exception as e:
        logger.error(f"Off-track question answer failed: {e}")
        return r("I'll answer that in a moment — let's finish your booking first.",
                  "Iska jawab thori dair mein dunga — pehle booking complete kar lein.",
                  "اس کا جواب جلد دوں گا — پہلے بکنگ مکمل کر لیں۔", lang)

# ── Phone formatter ────────────────────────────────────────────────────────────
def format_phone(phone: str) -> Optional[str]:
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 11 and digits.startswith('0'): return '+92' + digits[1:]
    if len(digits) == 12 and digits.startswith('92'): return '+' + digits
    if len(digits) == 10: return '+92' + digits
    if len(phone.strip()) > 0 and phone.strip()[0] == '+' and len(digits) >= 12: return '+' + digits
    return None

# ── Email validator ────────────────────────────────────────────────────────────
def is_valid_email(email: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', email.strip()))

# ── Time validator ─────────────────────────────────────────────────────────────
SHOP_OPEN = 10
SHOP_CLOSE = 20
TIME_SLOTS_24H = ["10:00", "11:00", "12:00", "13:00", "14:00",
                  "15:00", "16:00", "17:00", "18:00", "19:00"]

def format_time_display(time_24: str) -> str:
    """Convert internal 24-hour 'HH:MM' to friendly '1:00 PM' for messages only."""
    try:
        hour, mins = map(int, time_24.split(':'))
        display_hour = hour if hour <= 12 else hour - 12
        if display_hour == 0: display_hour = 12
        meridiem = "AM" if hour < 12 else "PM"
        return f"{display_hour}:{mins:02d} {meridiem}"
    except:
        return time_24

def parse_time(text: str) -> Optional[str]:
    text = text.lower().strip()
    # If it's already a clean 24-hour HH:MM (e.g. from a clicked time button), use it directly
    m24 = re.match(r'^(\d{1,2}):(\d{2})$', text)
    if m24:
        hour, mins = int(m24.group(1)), int(m24.group(2))
        if SHOP_OPEN <= hour < SHOP_CLOSE:
            return f"{hour:02d}:{mins:02d}"
        return None
    m = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', text)
    if not m: return None
    hour = int(m.group(1))
    mins = int(m.group(2)) if m.group(2) else 0
    meridiem = m.group(3)
    if meridiem == 'pm' and hour != 12: hour += 12
    if meridiem == 'am' and hour == 12: hour = 0
    if hour >= 7 and hour < 10 and not meridiem: hour += 12
    if not (SHOP_OPEN <= hour < SHOP_CLOSE): return None
    return f"{hour:02d}:{mins:02d}"

# ══════════════════════════════════════════════════════════════════════════════
# SLOT HELPERS — matches real table: id, Date, Time, Status, Booked By, Day, Phone, Booking ID
# ══════════════════════════════════════════════════════════════════════════════
def get_available_dates():
    """Returns list of distinct dates that have at least one Available slot."""
    try:
        res = supabase.table("slots").select("Date").eq("Status", "Available").order("Date").execute()
        seen = []
        for row in res.data:
            d = row.get("Date")
            if d and d not in seen:
                seen.append(d)
        return seen[:14]
    except: return []

def get_available_times(booking_date: str):
    """Returns list of available Time strings for a specific date."""
    try:
        res = supabase.table("slots").select("Time").eq("Date", booking_date).eq("Status", "Available").order("Time").execute()
        return [row.get("Time") for row in res.data if row.get("Time")]
    except: return []

def is_slot_available(booking_date: str, booking_time: str = None) -> bool:
    try:
        q = supabase.table("slots").select("*").eq("Date", booking_date).eq("Status", "Available")
        if booking_time:
            q = q.eq("Time", booking_time)
        res = q.execute()
        return len(res.data) > 0
    except: return True

def build_slot_buttons(dates: list) -> list:
    return dates

def build_time_buttons(times: list) -> list:
    """Returns list of {label, value} dicts: label is 12-hour for display,
    value is the original 24-hour string that matches Supabase exactly."""
    result = []
    for t in times:
        result.append({"label": format_time_display(t), "value": t})
    return result

def book_slot(booking_date: str, booking_time: str, phone: str, booking_id: str) -> bool:
    """Marks a specific Date+Time slot as Booked and links it to the booking."""
    try:
        res = supabase.table("slots").update({
            "Status": "Booked",
            "Booked By": phone,
            "Phone": phone,
            "Booking ID": booking_id,
        }).eq("Date", booking_date).eq("Time", booking_time).eq("Status", "Available").execute()
        return len(res.data) > 0
    except: return False

def free_slot_by_booking_id(booking_id: str):
    """Frees up a slot when a booking is cancelled or rescheduled away from it."""
    try:
        supabase.table("slots").update({
            "Status": "Available",
            "Booked By": "EMPTY",
            "Phone": "EMPTY",
            "Booking ID": "",
        }).eq("Booking ID", booking_id).execute()
    except: pass

# ── Duplicate booking check ────────────────────────────────────────────────────
def has_duplicate_booking(phone: str, booking_date: str) -> bool:
    try:
        res = supabase.table("bookings").select("*").eq("Phone", phone).eq("Date", booking_date).execute()
        return len(res.data) > 0
    except: return False

# ── Find booking by phone ──────────────────────────────────────────────────────
def find_bookings_by_phone(phone: str) -> list:
    try:
        today = str(date.today())
        res = supabase.table("bookings").select("*")\
            .eq("Phone", phone)\
            .gte("Date", today)\
            .neq("Status", "Cancelled")\
            .order("Date")\
            .execute()
        return res.data or []
    except: return []

def find_booking_by_phone(phone: str) -> Optional[dict]:
    bookings = find_bookings_by_phone(phone)
    return bookings[0] if bookings else None

def cancel_booking_by_id(booking_id: str):
    try:
        supabase.table("bookings").update({"Status": "Cancelled"}).eq("Booking ID", booking_id).execute()
        free_slot_by_booking_id(booking_id)
    except: pass

# ── Session manager ────────────────────────────────────────────────────────────
# ── Persistent session manager (Supabase-backed) ───────────────────────────────
# Replaces the in-memory dict so sessions survive Railway redeploys/restarts.
# In-process cache avoids hitting Supabase on every single field read within
# one request — we still load fresh at the start of each request and save
# once at the end.
_session_cache: Dict[str, dict] = {}

def get_session(session_id: str) -> dict:
    if session_id in _session_cache:
        return _session_cache[session_id]
    try:
        res = supabase.table("chat_sessions").select("*").eq("session_id", session_id).execute()
        if res.data:
            row = res.data[0]
            session = {
                "step": row.get("step") or "idle",
                "language": row.get("language"),
                "mode": row.get("mode"),
                "collected": row.get("collected") or {},
                "history": row.get("history") or [],
                "booking_id": row.get("booking_id"),
            }
        else:
            session = {"step": "idle", "language": None, "mode": None, "collected": {}, "history": [], "booking_id": None}
    except Exception:
        # Supabase unreachable — fall back to a fresh in-memory session rather than crashing
        session = {"step": "idle", "language": None, "mode": None, "collected": {}, "history": [], "booking_id": None}
    _session_cache[session_id] = session
    return session

def save_session(session_id: str, session: dict):
    _session_cache[session_id] = session
    try:
        payload = {
            "session_id": session_id,
            "step": session.get("step", "idle"),
            "language": session.get("language"),
            "mode": session.get("mode"),
            "collected": session.get("collected", {}),
            "history": session.get("history", [])[-20:],  # cap history size stored
            "updated_at": datetime.utcnow().isoformat(),
        }
        # Persist booking link when present (nullable column on chat_sessions)
        if session.get("booking_id"):
            payload["booking_id"] = session["booking_id"]
        supabase.table("chat_sessions").upsert(payload).execute()
    except Exception:
        pass  # don't break the chat response if persistence fails — cache still has it for this process

def reset_session(session_id: str):
    # Reset flow state for the next turn, but keep history (+ booking_id) so
    # owners can review the conversation after a booking is confirmed.
    existing = _session_cache.get(session_id) or {}
    fresh = {
        "step": "idle",
        "language": None,
        "mode": None,
        "collected": {},
        "history": existing.get("history") or [],
        "booking_id": existing.get("booking_id"),
    }
    _session_cache[session_id] = fresh
    # Intentionally do NOT delete the chat_sessions row.

# ── Save lead on exit ──────────────────────────────────────────────────────────
def save_lead(collected: dict):
    try:
        if collected.get("phone") or collected.get("name"):
            supabase.table("leads").insert({
                "Name": collected.get("name", "Unknown"),
                "Phone": collected.get("phone", ""),
                "Device": collected.get("device", ""),
                "Issue": collected.get("issue", ""),
            }).execute()
    except: pass

# ── Language-aware short replies ───────────────────────────────────────────────
def r(english: str, roman: str, urdu: str, lang: str) -> str:
    if lang == 'urdu': return urdu
    if lang == 'roman_urdu': return roman
    return english

# ── Owner context builder ──────────────────────────────────────────────────────
def build_owner_context():
    try:
        bookings = supabase.table("bookings").select("*").order("Date", desc=True).limit(50).execute().data
        slots = supabase.table("slots").select("*").eq("Status", "Available").order("Date").limit(30).execute().data
        leads = supabase.table("leads").select("*").order("created_at", desc=True).limit(20).execute().data
        today = str(date.today())
        upcoming = [b for b in bookings if (b.get("Date") or "")[:10] >= today]
        unpaid = [b for b in bookings if (b.get("Payment Status") or "").lower() == "unpaid"]

        slots_by_date = {}
        for s in slots:
            d = s.get("Date")
            slots_by_date.setdefault(d, []).append(s.get("Time"))

        context = f"""
=== FixPro iPhone Repair — Live Data (as of {today}) ===

UPCOMING BOOKINGS ({len(upcoming)} total):
{chr(10).join([f"- {b.get('Date')} {b.get('Time')} | {b.get('Name')} | {b.get('Phone')} | {b.get('Device')} | {b.get('Service')} | {b.get('Status')} | {b.get('Payment Status')}" for b in upcoming[:20]]) or "None"}

UNPAID BOOKINGS ({len(unpaid)} total):
{chr(10).join([f"- {b.get('Date')} | {b.get('Name')} | {b.get('Phone')} | {b.get('Device')}" for b in unpaid[:10]]) or "None"}

SLOT AVAILABILITY:
{chr(10).join([f"- {d}: {len(times)} slots available ({', '.join(times)})" for d, times in list(slots_by_date.items())[:10]]) or "No slots available"}

LEADS ({len(leads)} total, last 10):
{chr(10).join([f"- {l.get('Name')} | {l.get('Phone')} | {l.get('Device')} | {l.get('Issue')}" for l in leads[:10]]) or "None"}
""".strip()
        return context
    except Exception as e:
        return f"Error fetching data: {str(e)}"

# ══════════════════════════════════════════════════════════════════════════════
# OWNER CHAT
# ══════════════════════════════════════════════════════════════════════════════
# ── Owner: list all chat sessions (for dashboard Chats page) ───────────────────
@router.get("/sessions")
def list_chat_sessions(user=Depends(verify_token), limit: int = 100):
    """Return recent chat_sessions for the owner dashboard (read-only)."""
    try:
        cap = max(1, min(int(limit or 100), 200))
        res = (
            supabase.table("chat_sessions")
            .select("session_id, collected, history, updated_at, booking_id")
            .order("updated_at", desc=True)
            .limit(cap)
            .execute()
        )
        return res.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/owner")
def owner_chat(req: OwnerChatRequest, user=Depends(verify_token)):
    try:
        if req.context:
            bookings = req.context.get("bookings", [])
            leads = req.context.get("leads", [])
            revenue = req.context.get("revenue", 0)
            today = str(date.today())
            context = f"""
=== FixPro iPhone Repair — Live Data (as of {today}) ===

BOOKINGS ({len(bookings)} total):
{chr(10).join([f"- {b.get('Date')} {b.get('Time')} | {b.get('Name')} | {b.get('Phone')} | {b.get('Device')} | {b.get('Service')} | {b.get('Status')} | {b.get('Payment Status')}" for b in bookings[:30]]) or "None"}

REVENUE (confirmed bookings): Rs{revenue:,}

LEADS ({len(leads)} total):
{chr(10).join([f"- {l.get('Name')} | {l.get('Phone')} | {l.get('Device')} | {l.get('Issue')}" for l in leads[:10]]) or "None"}
""".strip()
        else:
            context = build_owner_context()

        system_prompt = f"""You are FixPro Assistant — the smartest employee at FixPro iPhone Repair in Lahore.

RULES:
- You have REAL live shop data below. ALWAYS use it to answer. Never say you don't have access.
- Give SPECIFIC answers using actual names, numbers, dates from the data.
- If owner asks "who hasn't paid" — list actual names and phones.
- If owner asks "today's bookings" — list them with time, name, device.
- Be concise and direct. No fluff.
- Language: Roman Urdu input → Roman Urdu reply. English input → English reply.

{context}"""

        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": m.role, "content": m.content} for m in req.messages]
        res = groq_client.chat.completions.create(model=GROQ_MODEL, messages=messages, max_tokens=600, temperature=0.4, timeout=15)
        return {"reply": res.choices[0].message.content}
    except Exception as e:
        logger.error(f"Owner chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOMER CHAT
# ══════════════════════════════════════════════════════════════════════════════
@router.post("/customer")
def customer_chat(req: CustomerChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    if is_rate_limited(session_id):
        return respond(
            "You're sending messages a bit fast! Please wait a moment and try again. 😊",
            session_id, typing_delay_ms=0
        )

    session = get_session(session_id)
    logger.info(f"[{session_id}] step={session.get('step')} msg={req.message[:80]!r}")
    try:
        result = _handle_customer_message(req, session_id, session)
        save_session(session_id, session)
        logger.info(f"[{session_id}] -> step={session.get('step')}")
        return result
    except HTTPException:
        save_session(session_id, session)
        raise
    except Exception as e:
        logger.error(f"[{session_id}] Unhandled error: {e}")
        save_session(session_id, session)
        return respond("Sorry, something went wrong. Please try again.", session_id, typing_delay_ms=0)


def _handle_customer_message(req: CustomerChatRequest, session_id: str, session: dict):
        msg = req.message.strip()

        if is_spam(msg):
            return respond(
                "Please keep the conversation respectful 😊 How can I help you book a repair?",
                session_id
            )

        if not session["language"]:
            session["language"] = detect_language(msg)
        lang = session["language"]

        step = session["step"]
        collected = session["collected"]
        mode = session.get("mode")

        if is_exit(msg) and step not in ["confirm", "confirm_cancel", "confirm_reschedule"]:
            save_lead(collected)
            reset_session(session_id)
            return respond(
                r("No problem! We've saved your info and will reach out soon. Take care! 👋",
                  "Koi baat nahi! Info save kar li. Allah Hafiz! 👋",
                  "کوئی بات نہیں! معلومات محفوظ کر لی۔ اللہ حافظ! 👋", lang),
                session_id, session_reset=True
            )

        if is_reschedule(msg) and step not in ["get_reschedule_phone", "get_new_date", "get_new_time", "confirm_reschedule"]:
            session["mode"] = "reschedule"
            session["step"] = "get_reschedule_phone"
            session["collected"] = {}
            return respond(
                r("Sure, I can help you reschedule! 📅\n\nPlease share the phone number used for your booking.",
                  "Bilkul, main reschedule mein madad karta hun! 📅\n\nApni booking wala phone number dijiye.",
                  "ضرور، میں دوبارہ شیڈول میں مدد کروں گا! 📅\n\nاپنی بکنگ والا فون نمبر دیں۔", lang),
                session_id
            )

        if is_cancel_booking(msg) and step not in ["get_cancel_phone", "confirm_cancel"]:
            session["mode"] = "cancel"
            session["step"] = "get_cancel_phone"
            session["collected"] = {}
            return respond(
                r("I can help you cancel your booking. 😔\n\nPlease share the phone number used for your booking.",
                  "Main aapki booking cancel karne mein madad karta hun. 😔\n\nBooking wala phone number dijiye.",
                  "میں آپ کی بکنگ منسوخ کرنے میں مدد کروں گا۔ 😔\n\nبکنگ والا فون نمبر دیں۔", lang),
                session_id
            )

        session["history"].append({"role": "user", "content": msg})

        # ── Mid-flow question interception ─────────────────────────────────────
        # If user asks an unrelated question while we're collecting booking info,
        # answer it via RAG, then re-ask the SAME field instead of breaking the flow.
        DATA_COLLECTION_STEPS = ["get_device", "get_date", "get_time", "get_name",
                                  "get_phone", "get_email", "get_new_date", "get_new_time"]
        REPROMPT_TEXT = {
            "get_device": r("Now, what device do you have and what's the issue?",
                             "Ab batayein, kaun sa device hai aur kya masla hai?",
                             "اب بتائیں، کون سا ڈیوائس ہے اور کیا مسئلہ ہے؟", lang),
            "get_date": r("Which date works for your appointment?",
                          "Appointment ke liye kaun si date theek hai?",
                          "ملاقات کے لیے کون سی تاریخ مناسب ہے؟", lang),
            "get_time": r("What time would you prefer?",
                         "Kaun sa waqt chahiye?",
                         "کیا وقت مناسب ہے؟", lang),
            "get_name": r("What's your full name?", "Aapka poora naam?", "آپ کا پورا نام؟", lang),
            "get_phone": r("What's your phone number?", "Phone number?", "فون نمبر؟", lang),
            "get_email": r("What's your email? (or type 'skip')", "Email? (ya 'skip' likhein)", "ای میل؟ (یا 'skip' لکھیں)", lang),
            "get_new_date": r("Which new date would you like?", "Nayi date batayein?", "نئی تاریخ بتائیں؟", lang),
            "get_new_time": r("What new time would you prefer?", "Naya waqt batayein?", "نیا وقت بتائیں؟", lang),
        }
        if step in DATA_COLLECTION_STEPS and looks_like_question(msg, step):
            answer = answer_offtrack_question(msg, lang, session["history"])
            reprompt = REPROMPT_TEXT.get(step, "")
            combined = f"{answer}\n\n{reprompt}"
            session["history"].append({"role": "assistant", "content": combined})
            # Re-show relevant buttons if applicable
            if step in ("get_date", "get_new_date"):
                return respond(combined, session_id, slot_buttons=build_slot_buttons(get_available_dates()))
            if step in ("get_time", "get_new_time"):
                d = collected.get("date") or collected.get("new_date") or ""
                times = get_available_times(d)
                return respond(combined, session_id, time_buttons=build_time_buttons(times) or build_time_buttons(TIME_SLOTS_24H))
            return respond(combined, session_id)

        # ════════════════════════════════════════════════════════════════════
        # CANCEL FLOW
        # ════════════════════════════════════════════════════════════════════
        if step == "get_cancel_phone":
            formatted = format_phone(msg)
            if not formatted:
                return respond(
                    r("Please enter a valid phone number (e.g. 03001234567)",
                      "Sahi phone number dalein (maslan: 03001234567)",
                      "درست فون نمبر درج کریں", lang),
                    session_id
                )
            bookings = find_bookings_by_phone(formatted)
            if not bookings:
                reset_session(session_id)
                return respond(
                    r("No upcoming booking found for that number. Please check and try again, or call us at +92 300 1234567.",
                      "Is number pe koi upcoming booking nahi mili. Dobara check karein ya +92 300 1234567 pe call karein.",
                      "اس نمبر پر کوئی آنے والی بکنگ نہیں ملی۔", lang),
                    session_id, session_reset=True
                )
            booking = bookings[0]
            note = ""
            if len(bookings) > 1:
                note = r(f"\n\n(You have {len(bookings)} upcoming bookings — showing the earliest. Call us if you meant a different one.)",
                         f"\n\n(Aapki {len(bookings)} bookings hain — sab se pehli dikha rahe hain.)",
                         f"\n\n({len(bookings)} بکنگز ہیں — پہلی دکھا رہے ہیں۔)", lang)
            collected["booking_to_cancel"] = booking
            collected["phone"] = formatted
            session["step"] = "confirm_cancel"
            return respond(
                r(f"Found your booking:\n\n"
                  f"📅 Date: {booking.get('Date')}\n"
                  f"⏰ Time: {booking.get('Time')}\n"
                  f"📱 Device: {booking.get('Device')}\n"
                  f"🔧 Issue: {booking.get('Issue')}\n\n"
                  f"Are you sure you want to cancel? Type YES to confirm or NO to keep it.{note}",
                  f"Aapki booking mili:\n\n"
                  f"📅 Date: {booking.get('Date')}\n"
                  f"⏰ Waqt: {booking.get('Time')}\n"
                  f"📱 Device: {booking.get('Device')}\n\n"
                  f"Kya aap wakai cancel karna chahte hain? YES ya NO likhein.{note}",
                  f"آپ کی بکنگ ملی:\n\n"
                  f"📅 تاریخ: {booking.get('Date')}\n"
                  f"⏰ وقت: {booking.get('Time')}\n\n"
                  f"کیا آپ واقعی منسوخ کرنا چاہتے ہیں؟ YES یا NO لکھیں۔{note}", lang),
                session_id,
                quick_replies=["YES, Cancel", "NO, Keep it"]
            )

        if step == "confirm_cancel":
            if is_yes(msg):
                booking = collected.get("booking_to_cancel", {})
                cancel_booking_by_id(booking.get("Booking ID", ""))
                reset_session(session_id)
                return respond(
                    r(f"✅ Your booking for {booking.get('Date')} at {booking.get('Time')} has been cancelled.\n\nWe hope to see you again soon! Book again anytime. 😊",
                      f"✅ Aapki {booking.get('Date')} ko {booking.get('Time')} baje ki booking cancel ho gayi.\n\nUmeed hai jaldi milenge! 😊",
                      f"✅ آپ کی بکنگ منسوخ ہو گئی۔\n\nامید ہے جلد ملیں گے! 😊", lang),
                    session_id, session_reset=True,
                    quick_replies=["Book Again"]
                )
            else:
                reset_session(session_id)
                return respond(
                    r("No problem! Your booking is kept. See you on your appointment day! 🔧",
                      "Theek hai! Aapki booking safe hai. Appointment pe milenge! 🔧",
                      "ٹھیک ہے! آپ کی بکنگ محفوظ ہے۔ ملاقات پر ملیں گے! 🔧", lang),
                    session_id, session_reset=True
                )

        # ════════════════════════════════════════════════════════════════════
        # RESCHEDULE FLOW
        # ════════════════════════════════════════════════════════════════════
        if step == "get_reschedule_phone":
            formatted = format_phone(msg)
            if not formatted:
                return respond(
                    r("Please enter a valid phone number (e.g. 03001234567)",
                      "Sahi phone number dalein",
                      "درست فون نمبر درج کریں", lang),
                    session_id
                )
            bookings = find_bookings_by_phone(formatted)
            if not bookings:
                reset_session(session_id)
                return respond(
                    r("No upcoming booking found for that number. Please check and try again, or call us at +92 300 1234567.",
                      "Is number pe koi upcoming booking nahi mili. Dobara check karein.",
                      "اس نمبر پر کوئی بکنگ نہیں ملی۔", lang),
                    session_id, session_reset=True
                )
            booking = bookings[0]
            note = ""
            if len(bookings) > 1:
                note = r(f"\n\n(You have {len(bookings)} upcoming bookings — rescheduling the earliest. Call us if you meant a different one.)",
                         f"\n\n(Aapki {len(bookings)} bookings hain — sab se pehli reschedule kar rahe hain.)",
                         f"\n\n({len(bookings)} بکنگز ہیں — پہلی کو دوبارہ شیڈول کر رہے ہیں۔)", lang)
            collected["booking_to_reschedule"] = booking
            collected["phone"] = formatted
            session["step"] = "get_new_date"

            available_dates = get_available_dates()
            slot_btns = build_slot_buttons(available_dates)
            slots_text = "\n".join([f"• {d}" for d in available_dates[:7]]) or "Please call us"

            return respond(
                r(f"Found your booking on {booking.get('Date')} at {booking.get('Time')}.{note}\n\n"
                  f"Available dates:\n{slots_text}\n\nWhich new date would you like?",
                  f"Aapki booking mili: {booking.get('Date')} ko {booking.get('Time')} baje.{note}\n\n"
                  f"Available dates:\n{slots_text}\n\nNayi date batayein.",
                  f"آپ کی بکنگ ملی۔{note}\n\nدستیاب تاریخیں:\n{slots_text}\n\nنئی تاریخ بتائیں۔", lang),
                session_id, slot_buttons=slot_btns
            )

        if step == "get_new_date":
            parsed = parse_date(msg)
            if not parsed:
                slot_btns = build_slot_buttons(get_available_dates())
                return respond(
                    r("Couldn't understand that date. Please pick from the available dates or say 'tomorrow', 'Saturday' etc.",
                      "Date samajh nahi aayi. Available dates mein se chunein ya 'kal', 'Saturday' bolein.",
                      "تاریخ سمجھ نہیں آئی۔", lang),
                    session_id, slot_buttons=slot_btns
                )
            if parsed < str(date.today()):
                return respond(r("That date has passed! Choose a future date.", "Yeh date guzar gayi! Aagay ki date lo.", "یہ تاریخ گزر گئی!", lang), session_id)
            if not is_slot_available(parsed):
                slot_btns = build_slot_buttons(get_available_dates())
                return respond(
                    r(f"Sorry, {parsed} is fully booked! Please choose another date.",
                      f"Sorry, {parsed} full hai! Koi aur date lo.",
                      f"معذرت، {parsed} بھرا ہوا ہے!", lang),
                    session_id, slot_buttons=slot_btns
                )
            collected["new_date"] = parsed
            session["step"] = "get_new_time"
            times = get_available_times(parsed)
            time_btns = build_time_buttons(times) or build_time_buttons(TIME_SLOTS_24H)
            return respond(
                r(f"✅ {parsed} works!\n\nWhat time would you prefer? (Our hours: 10 AM – 8 PM)",
                  f"✅ {parsed} theek hai!\n\nKaun sa waqt chahiye? (10 AM – 8 PM)",
                  f"✅ {parsed} ٹھیک ہے!\n\nکیا وقت مناسب ہے؟ (10 بجے – 8 بجے)", lang),
                session_id, time_buttons=time_btns
            )

        if step == "get_new_time":
            parsed_time = parse_time(msg)
            if not parsed_time:
                times = get_available_times(collected.get("new_date", ""))
                return respond(
                    r("Please give a valid time between 10 AM and 8 PM (e.g. '2 PM', '4:30 PM')",
                      "10 AM se 8 PM ke beech ka waqt dijiye",
                      "10 بجے سے 8 بجے کے درمیان وقت دیں", lang),
                    session_id, time_buttons=build_time_buttons(times) or build_time_buttons(TIME_SLOTS_24H)
                )
            if not is_slot_available(collected.get("new_date", ""), parsed_time):
                times = get_available_times(collected.get("new_date", ""))
                return respond(
                    r("Sorry, that time just got booked! Please pick another time.",
                      "Sorry, yeh waqt abhi book ho gaya! Doosra waqt chunein.",
                      "معذرت، یہ وقت ابھی بک ہو گیا!", lang),
                    session_id, time_buttons=build_time_buttons(times) or build_time_buttons(TIME_SLOTS_24H)
                )
            collected["new_time"] = parsed_time
            session["step"] = "confirm_reschedule"
            booking = collected.get("booking_to_reschedule", {})
            return respond(
                r(f"Please confirm the reschedule:\n\n"
                  f"📅 Old: {booking.get('Date')} at {booking.get('Time')}\n"
                  f"📅 New: {collected['new_date']} at {parsed_time}\n\n"
                  f"Type YES to confirm or NO to cancel.",
                  f"Reschedule confirm karein:\n\n"
                  f"📅 Purana: {booking.get('Date')} - {booking.get('Time')}\n"
                  f"📅 Naya: {collected['new_date']} - {parsed_time}\n\n"
                  f"YES ya NO likhein.",
                  f"دوبارہ شیڈول کی تصدیق:\n\n"
                  f"📅 پرانا: {booking.get('Date')}\n"
                  f"📅 نیا: {collected['new_date']}\n\n"
                  f"YES یا NO لکھیں۔", lang),
                session_id, quick_replies=["YES, Reschedule", "NO, Keep original"]
            )

        if step == "confirm_reschedule":
            if is_yes(msg):
                booking = collected.get("booking_to_reschedule", {})
                old_booking_id = booking.get("Booking ID", "")
                try:
                    free_slot_by_booking_id(old_booking_id)
                    book_slot(collected["new_date"], collected["new_time"], collected.get("phone", ""), old_booking_id)
                    supabase.table("bookings").update({
                        "Date": collected["new_date"],
                        "Time": collected["new_time"],
                        "Status": "Pending",
                    }).eq("Booking ID", old_booking_id).execute()
                except Exception as e:
                    return respond(r("Sorry, reschedule failed. Please call us at +92 300 1234567.",
                                    "Sorry, reschedule nahi ho saka. +92 300 1234567 pe call karein.",
                                    "معذرت، دوبارہ شیڈول ناکام ہوا۔", lang), session_id)
                reset_session(session_id)
                return respond(
                    r(f"✅ Rescheduled! Your new appointment is on {collected['new_date']} at {collected['new_time']}.\n\nSee you then! 🔧",
                      f"✅ Reschedule ho gaya! Naya appointment: {collected['new_date']} - {collected['new_time']} baje.\n\nMilenge! 🔧",
                      f"✅ دوبارہ شیڈول ہو گیا! نئی ملاقات: {collected['new_date']}۔ 🔧", lang),
                    session_id, session_reset=True
                )
            else:
                reset_session(session_id)
                return respond(
                    r("No problem! Your original booking is unchanged. 😊",
                      "Theek hai! Aapka purana appointment safe hai. 😊",
                      "ٹھیک ہے! آپ کی اصل بکنگ محفوظ ہے۔ 😊", lang),
                    session_id, session_reset=True
                )

        # ════════════════════════════════════════════════════════════════════
        # IDLE — classify intent
        # ════════════════════════════════════════════════════════════════════
        if step == "idle":
            available_dates = get_available_dates()
            slots_text = "\n".join([f"- {d}" for d in available_dates[:7]]) or "Call us for availability"

            classify_prompt = f"""Classify this message as BOOKING or QUESTION.
BOOKING = user wants to book/schedule a repair
QUESTION = asking about services, prices, location, hours, warranty, or anything else
Reply with ONLY: BOOKING or QUESTION

Message: {msg}"""

            intent = safe_groq_call(
                [{"role": "user", "content": classify_prompt}],
                max_tokens=10, temperature=0,
                fallback="QUESTION"
            ).strip().upper()

            if "BOOKING" in intent:
                session["step"] = "get_device"
                session["mode"] = "booking"
                reply = r(
                    "Great! I'll help you book a repair. 🔧\n\nWhat device do you have and what's the issue? (e.g. iPhone 13, cracked screen)",
                    "Zabardast! Booking mein madad karta hun. 🔧\n\nKaun sa device hai aur kya masla hai? (maslan: iPhone 13, screen toot gayi)",
                    "بہت اچھا! بکنگ میں مدد کروں گا۔ 🔧\n\nکون سا ڈیوائس ہے اور کیا مسئلہ ہے؟", lang
                )
            else:
                system_prompt = f"""You are a helpful assistant for FixPro iPhone Repair in Lahore, Pakistan.
Answer the customer's question using the shop info below. Be helpful, concise, and friendly.
Reply in the same language as the customer (English, Roman Urdu, or Urdu).
Keep response under 120 words. End by asking if they'd like to book an appointment.

{SHOP_RAG}

AVAILABLE DATES:
{slots_text}"""
                history_msgs = [{"role": "system", "content": system_prompt}]
                history_msgs += session["history"][-6:]
                reply = safe_groq_call(
                    history_msgs, max_tokens=250, temperature=0.4,
                    fallback=r(
                        "I'm having a little trouble right now — please call us at +92 300 1234567, or try asking again in a moment.",
                        "Abhi thori dair ke liye masla aa raha hai — +92 300 1234567 pe call karein ya dobara koshish karein.",
                        "ابھی مسئلہ ہے — براہ کرم +92 300 1234567 پر کال کریں۔", lang
                    )
                )

            session["history"].append({"role": "assistant", "content": reply})
            return respond(reply, session_id,
                quick_replies=["Book a Repair", "Check Prices", "Shop Hours", "Reschedule", "Cancel Booking"] if len(session["history"]) <= 2 else []
            )

        # ════════════════════════════════════════════════════════════════════
        # GET DEVICE + ISSUE
        # ════════════════════════════════════════════════════════════════════
        elif step == "get_device":
            if len(msg) < 3:
                return respond(
                    r("Please tell me your device and issue (e.g. iPhone 14, battery draining fast)",
                      "Device aur masla batayein (maslan: iPhone 14, battery jaldi khatam)",
                      "ڈیوائس اور مسئلہ بتائیں", lang),
                    session_id
                )
            parts = msg.split(',', 1)
            collected["device"] = parts[0].strip()
            collected["issue"] = parts[1].strip() if len(parts) > 1 else msg
            wait_time = get_wait_time(collected["issue"])

            session["step"] = "get_date"
            available_dates = get_available_dates()
            slot_btns = build_slot_buttons(available_dates)
            slots_text = "\n".join([f"• {d}" for d in available_dates[:7]]) or "Please call us"

            reply = r(
                f"Got it! **{collected['device']}** — {collected['issue']}.\n⏱ Estimated repair time: {wait_time}\n\nAvailable dates:\n{slots_text}\n\nWhich date works? (or say 'tomorrow', 'Saturday' etc.)",
                f"Theek hai! **{collected['device']}** — {collected['issue']}.\n⏱ Repair time: {wait_time}\n\nAvailable dates:\n{slots_text}\n\nKaun si date theek hai?",
                f"ٹھیک ہے! ⏱ مرمت کا وقت: {wait_time}\n\nدستیاب تاریخیں:\n{slots_text}\n\nکون سی تاریخ مناسب ہے؟", lang
            )
            session["history"].append({"role": "assistant", "content": reply})
            return respond(reply, session_id, slot_buttons=slot_btns)

        # ════════════════════════════════════════════════════════════════════
        # GET DATE
        # ════════════════════════════════════════════════════════════════════
        elif step == "get_date":
            parsed = parse_date(msg)
            if not parsed:
                slot_btns = build_slot_buttons(get_available_dates())
                return respond(
                    r("Couldn't understand that date. Pick from above or say 'tomorrow', 'Saturday', etc.",
                      "Date samajh nahi aayi. 'Kal', 'Saturday' ya upar se chunein.",
                      "تاریخ سمجھ نہیں آئی۔", lang),
                    session_id, slot_buttons=slot_btns
                )
            if parsed < str(date.today()):
                return respond(r("That date has passed! Please choose a future date.", "Yeh date guzar gayi! Aagay ki date lo.", "یہ تاریخ گزر گئی!", lang), session_id)
            if not is_slot_available(parsed):
                slot_btns = build_slot_buttons(get_available_dates())
                return respond(
                    r(f"Sorry, {parsed} is fully booked! Please choose another date.",
                      f"Sorry, {parsed} full hai! Koi aur date lo.",
                      f"معذرت، {parsed} بھرا ہوا ہے!", lang),
                    session_id, slot_buttons=slot_btns
                )
            collected["date"] = parsed
            session["step"] = "get_time"
            times = get_available_times(parsed)
            time_btns = build_time_buttons(times) or build_time_buttons(TIME_SLOTS_24H)
            reply = r(
                f"✅ {parsed} works!\n\nWhat time do you prefer? Our hours: 10 AM – 8 PM",
                f"✅ {parsed} theek hai!\n\nKaun sa waqt chahiye? Hum 10 AM – 8 PM tak khule hain.",
                f"✅ {parsed} ٹھیک ہے!\n\nکیا وقت مناسب ہے؟ ہم 10 بجے سے 8 بجے تک کھلے ہیں۔", lang
            )
            session["history"].append({"role": "assistant", "content": reply})
            return respond(reply, session_id, time_buttons=time_btns)

        # ════════════════════════════════════════════════════════════════════
        # GET TIME
        # ════════════════════════════════════════════════════════════════════
        elif step == "get_time":
            parsed_time = parse_time(msg)
            if not parsed_time:
                times = get_available_times(collected.get("date", ""))
                return respond(
                    r("Please give a valid time between 10 AM and 8 PM (e.g. '2 PM', '4:30 PM')",
                      "10 AM se 8 PM ke beech waqt dijiye",
                      "10 بجے سے 8 بجے کے درمیان وقت دیں", lang),
                    session_id, time_buttons=build_time_buttons(times) or build_time_buttons(TIME_SLOTS_24H)
                )
            if not is_slot_available(collected.get("date", ""), parsed_time):
                times = get_available_times(collected.get("date", ""))
                return respond(
                    r("Sorry, that time just got booked! Please pick another time.",
                      "Sorry, yeh waqt abhi book ho gaya! Doosra waqt chunein.",
                      "معذرت، یہ وقت ابھی بک ہو گیا!", lang),
                    session_id, time_buttons=build_time_buttons(times) or build_time_buttons(TIME_SLOTS_24H)
                )
            collected["time"] = parsed_time
            session["step"] = "get_name"
            reply = r("Perfect! Now, what's your full name?", "Zabardast! Aapka poora naam?", "بہت اچھا! آپ کا پورا نام؟", lang)
            session["history"].append({"role": "assistant", "content": reply})
            return respond(reply, session_id)

        # ════════════════════════════════════════════════════════════════════
        # GET NAME
        # ════════════════════════════════════════════════════════════════════
        elif step == "get_name":
            if len(msg.strip()) < 2 or re.match(r'^\d+$', msg.strip()):
                return respond(r("Please enter a valid name.", "Sahi naam dijiye.", "درست نام درج کریں۔", lang), session_id)
            collected["name"] = msg.strip().title()
            session["step"] = "get_phone"
            reply = r(
                "Got it! What's your phone number? (e.g. +923001234567 or 03001234567)",
                "Theek hai! Phone number? (maslan: 03001234567)",
                "فون نمبر؟ (مثلاً: 03001234567)", lang
            )
            session["history"].append({"role": "assistant", "content": reply})
            return respond(reply, session_id)

        # ════════════════════════════════════════════════════════════════════
        # GET PHONE
        # ════════════════════════════════════════════════════════════════════
        elif step == "get_phone":
            formatted = format_phone(msg)
            if not formatted:
                collected["phone_retries"] = collected.get("phone_retries", 0) + 1
                if collected["phone_retries"] >= 3:
                    save_lead(collected)
                    reset_session(session_id)
                    return respond(
                        r("Having trouble with the phone format. Please call us directly at +92 300 1234567 and we'll book you right away!",
                          "Phone number mein masla aa raha hai. Seedha +92 300 1234567 pe call karein, hum turant book kar dein ge!",
                          "فون نمبر میں مسئلہ ہے۔ براہ راست +92 300 1234567 پر کال کریں!", lang),
                        session_id, session_reset=True
                    )
                return respond(
                    r("Invalid phone number. Please enter with country code (e.g. 03001234567)",
                      "Phone number sahi nahi. Country code ke saath dalein (03001234567)",
                      "فون نمبر درست نہیں۔", lang),
                    session_id
                )
            if has_duplicate_booking(formatted, collected.get("date", "")):
                session["step"] = "get_date"
                slot_btns = build_slot_buttons(get_available_dates())
                return respond(
                    r(f"You already have a booking on {collected.get('date')}! Please choose a different date.",
                      f"Aapki {collected.get('date')} ko pehle se booking hai! Aur date lo.",
                      f"آپ کی {collected.get('date')} کو پہلے سے بکنگ ہے!", lang),
                    session_id, slot_buttons=slot_btns
                )
            collected["phone"] = formatted
            session["step"] = "get_email"
            reply = r(
                "What's your email? (for booking confirmation — type 'skip' to skip)",
                "Email address? (booking confirmation ke liye — 'skip' likh sakte hain)",
                "ای میل؟ (بکنگ کنفرمیشن کے لیے — 'skip' لکھ سکتے ہیں)", lang
            )
            session["history"].append({"role": "assistant", "content": reply})
            return respond(reply, session_id)

        # ════════════════════════════════════════════════════════════════════
        # GET EMAIL
        # ════════════════════════════════════════════════════════════════════
        elif step == "get_email":
            if is_skip(msg):
                collected["email"] = ""
            elif not is_valid_email(msg):
                return respond(
                    r("Invalid email. Please enter a valid email (e.g. name@gmail.com) or type 'skip'",
                      "Email sahi nahi. Sahi email dalein ya 'skip' likhein",
                      "ای میل درست نہیں۔ درست ای میل درج کریں یا 'skip' لکھیں", lang),
                    session_id
                )
            else:
                collected["email"] = msg.strip().lower()

            session["step"] = "confirm"
            wait_time = get_wait_time(collected.get("issue", ""))
            reply = r(
                f"Please confirm your booking:\n\n"
                f"📱 Device: {collected.get('device')}\n"
                f"🔧 Issue: {collected.get('issue')}\n"
                f"⏱ Est. time: {wait_time}\n"
                f"📅 Date: {collected.get('date')}\n"
                f"⏰ Time: {collected.get('time')}\n"
                f"👤 Name: {collected.get('name')}\n"
                f"📞 Phone: {collected.get('phone')}\n"
                f"{'📧 Email: ' + collected.get('email') if collected.get('email') else ''}\n\n"
                f"Type YES to confirm or NO to cancel.",

                f"Booking confirm karein:\n\n"
                f"📱 Device: {collected.get('device')}\n"
                f"🔧 Masla: {collected.get('issue')}\n"
                f"⏱ Repair time: {wait_time}\n"
                f"📅 Date: {collected.get('date')}\n"
                f"⏰ Waqt: {collected.get('time')}\n"
                f"👤 Naam: {collected.get('name')}\n"
                f"📞 Phone: {collected.get('phone')}\n\n"
                f"YES ya NO likhein.",

                f"بکنگ کی تصدیق:\n\n"
                f"📱 ڈیوائس: {collected.get('device')}\n"
                f"🔧 مسئلہ: {collected.get('issue')}\n"
                f"⏱ وقت: {wait_time}\n"
                f"📅 تاریخ: {collected.get('date')}\n"
                f"⏰ وقت: {collected.get('time')}\n"
                f"👤 نام: {collected.get('name')}\n"
                f"📞 فون: {collected.get('phone')}\n\n"
                f"تصدیق کے لیے YES لکھیں۔", lang
            )
            session["history"].append({"role": "assistant", "content": reply})
            return respond(reply, session_id, quick_replies=["YES ✅", "NO ❌"])

        # ════════════════════════════════════════════════════════════════════
        # CONFIRM BOOKING
        # ════════════════════════════════════════════════════════════════════
        elif step == "confirm":
            if is_yes(msg):
                if not is_slot_available(collected.get("date", ""), collected.get("time", "")):
                    session["step"] = "get_date"
                    slot_btns = build_slot_buttons(get_available_dates())
                    return respond(
                        r("Sorry! That slot just got filled. Please choose another date.",
                          "Sorry! Slot abhi full ho gaya. Koi aur date lo.",
                          "معذرت! سلاٹ ابھی بھر گیا۔", lang),
                        session_id, slot_buttons=slot_btns
                    )
                booking_id = f"CUST-{uuid.uuid4().hex[:8].upper()}"
                try:
                    supabase.table("bookings").insert({
                        "Booking ID": booking_id,
                        "Name": collected.get("name", ""),
                        "Phone": collected.get("phone", ""),
                        "Email": collected.get("email", ""),
                        "Device": collected.get("device", ""),
                        "Issue": collected.get("issue", ""),
                        "Service": collected.get("issue", ""),
                        "Date": collected.get("date", ""),
                        "Time": collected.get("time", ""),
                        "Status": "Pending",
                        "Payment Status": "Unpaid",
                        "Notes": "Booked via customer chatbot",
                    }).execute()
                    book_slot(collected.get("date", ""), collected.get("time", ""), collected.get("phone", ""), booking_id)
                    logger.info(f"Booking created: {booking_id} | {collected.get('phone')} | {collected.get('date')} {collected.get('time')}")
                except Exception as e:
                    logger.error(f"Booking insert failed for {collected.get('phone')}: {e}")
                    save_lead(collected)
                    return respond(
                        r("Sorry, something went wrong saving your booking. We've noted your details — please call us at +92 300 1234567 to confirm.",
                          "Sorry, booking save karte waqt masla aa gaya. Aapki details note kar li hain — +92 300 1234567 pe call karein confirm karne ke liye.",
                          "معذرت، بکنگ محفوظ کرتے ہوئے مسئلہ ہوا۔ براہ کرم +92 300 1234567 پر کال کریں۔", lang),
                        session_id
                    )

                booking_info = {**collected, "booking_id": booking_id}
                session["booking_id"] = booking_id
                reset_session(session_id)
                reply = r(
                    f"🎉 Booking confirmed! Your ID: **{booking_id}**\n\nSee you on {booking_info.get('date')} at {booking_info.get('time')}. Please arrive 5 mins early.\n\nWould you mind leaving us a Google review after your repair? It helps us a lot! ⭐",
                    f"🎉 Booking confirm! ID: **{booking_id}**\n\n{booking_info.get('date')} ko {booking_info.get('time')} baje milenge. 5 minute pehle aa jayein.\n\nRepair ke baad Google review de dein — bohat madad hoti hai! ⭐",
                    f"🎉 بکنگ کنفرم! ID: **{booking_id}**\n\n{booking_info.get('date')} کو ملیں گے۔\n\nمرمت کے بعد Google review دیں! ⭐", lang
                )
                return respond(reply, session_id,
                    booking_created=True, booking_info=booking_info,
                    session_reset=True, rating_prompt=True,
                    quick_replies=["⭐ Leave a Review", "Book Another"]
                )

            elif is_no(msg):
                reset_session(session_id)
                return respond(
                    r("Booking cancelled. Feel free to start again anytime! 😊",
                      "Booking cancel. Jab chahein dobara shuru karein! 😊",
                      "بکنگ منسوخ۔ جب چاہیں دوبارہ شروع کریں! 😊", lang),
                    session_id, session_reset=True,
                    quick_replies=["Start Over"]
                )
            else:
                return respond(
                    r("Please type YES to confirm or NO to cancel.",
                      "YES ya NO likhein.",
                      "YES یا NO لکھیں۔", lang),
                    session_id, quick_replies=["YES ✅", "NO ❌"]
                )

        # Fallback
        return respond(
            r("How can I help you? 😊", "Main kaise madad kar sakta hun? 😊", "میں کیسے مدد کر سکتا ہوں؟ 😊", lang),
            session_id,
            quick_replies=["Book a Repair", "Check Prices", "Shop Hours", "Reschedule", "Cancel Booking"]
        )