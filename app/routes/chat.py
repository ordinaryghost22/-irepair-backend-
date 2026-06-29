from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict
from groq import Groq
from supabase import create_client
from app.config import SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY
from app.auth import verify_token
import uuid
import re
from datetime import date, timedelta, datetime

router = APIRouter()
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── In-memory session store ────────────────────────────────────────────────────
sessions: Dict[str, dict] = {}

# ── Models ─────────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class OwnerChatRequest(BaseModel):
    messages: List[Message]
    context: Optional[dict] = None

class CustomerChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None   # ← add this
    history: Optional[List[Message]] = []

# ── Profanity / spam filter ────────────────────────────────────────────────────
SPAM_PATTERNS = [
    r'\b(fuck|shit|bastard|asshole|bitch|harami|gandu|madarchod|benchod)\b',
]
def is_spam(text: str) -> bool:
    t = text.lower()
    for p in SPAM_PATTERNS:
        if re.search(p, t): return True
    if len(text) > 500: return True
    if re.match(r'^[^a-zA-Z\u0600-\u06FF\s]{10,}$', text): return True
    return False

# ── Exit intent detection ──────────────────────────────────────────────────────
EXIT_WORDS = ['cancel', 'never mind', 'nevermind', 'bye', 'goodbye', 'chor do', 'rehne do', 'nahi chahiye', 'band karo']
def is_exit(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in EXIT_WORDS)

# ── Language detection ─────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    urdu_chars = re.findall(r'[\u0600-\u06FF]', text)
    if len(urdu_chars) > 2: return 'urdu'
    roman_urdu = ['kya', 'hai', 'mera', 'meri', 'aap', 'hum', 'yeh', 'woh', 'nahi', 'haan', 'theek', 'kal', 'aj', 'aaj', 'phone', 'naam', 'karo', 'bhai', 'yaar', 'ap', 'ho', 'kar']
    words = text.lower().split()
    if sum(1 for w in words if w in roman_urdu) >= 2: return 'roman_urdu'
    return 'english'

# ── Smart date parser ──────────────────────────────────────────────────────────
def parse_date(text: str) -> Optional[str]:
    text = text.lower().strip()
    today = date.today()

    if any(w in text for w in ['today', 'aaj', 'aj']): return str(today)
    if any(w in text for w in ['tomorrow', 'kal', 'kl']): return str(today + timedelta(days=1))
    if any(w in text for w in ['day after tomorrow', 'parso', 'parsoon']): return str(today + timedelta(days=2))

    days = {
        'monday': 0, 'somwar': 0,
        'tuesday': 1, 'mangal': 1,
        'wednesday': 2, 'budh': 2,
        'thursday': 3, 'jumeraat': 3, 'jumerat': 3,
        'friday': 4, 'jumma': 4, 'juma': 4,
        'saturday': 5, 'hafta': 5,
        'sunday': 6, 'itwar': 6, 'itrawar': 6,
    }
    for day_name, day_num in days.items():
        if day_name in text:
            days_ahead = day_num - today.weekday()
            if days_ahead <= 0: days_ahead += 7
            return str(today + timedelta(days=days_ahead))

    # Try direct date formats
    patterns = [
        r'(\d{4}-\d{2}-\d{2})',
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',
        r'(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                if '-' in m.group(0) and len(m.group(0)) == 10:
                    d = datetime.strptime(m.group(0), '%Y-%m-%d').date()
                    if d >= today: return str(d)
            except: pass

    return None

# ── Phone formatter ────────────────────────────────────────────────────────────
def format_phone(phone: str) -> Optional[str]:
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 11 and digits.startswith('0'):
        return '+92' + digits[1:]
    if len(digits) == 12 and digits.startswith('92'):
        return '+' + digits
    if len(digits) == 10:
        return '+92' + digits
    if digits.startswith('+'):
        if len(digits) >= 12: return '+' + digits
    return None

# ── Email validator ────────────────────────────────────────────────────────────
def is_valid_email(email: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', email.strip()))

# ── Time validator ─────────────────────────────────────────────────────────────
SHOP_OPEN = 10   # 10 AM
SHOP_CLOSE = 20  # 8 PM

def parse_time(text: str) -> Optional[str]:
    text = text.lower().strip()
    m = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', text)
    if not m: return None
    hour = int(m.group(1))
    mins = int(m.group(2)) if m.group(2) else 0
    meridiem = m.group(3)
    if meridiem == 'pm' and hour != 12: hour += 12
    if meridiem == 'am' and hour == 12: hour = 0
    if not (SHOP_OPEN <= hour < SHOP_CLOSE): return None
    return f"{hour:02d}:{mins:02d}"

# ── Duplicate booking check ────────────────────────────────────────────────────
def has_duplicate_booking(phone: str, booking_date: str) -> bool:
    try:
        res = supabase.table("bookings").select("*").eq("Phone", phone).eq("Date", booking_date).execute()
        return len(res.data) > 0
    except: return False

# ── Slot availability check ────────────────────────────────────────────────────
def get_available_slots():
    try:
        res = supabase.table("slots").select("*").order("Date").limit(14).execute()
        return [s for s in res.data if (s.get("available") or 0) > 0]
    except: return []

def is_slot_available(booking_date: str) -> bool:
    try:
        res = supabase.table("slots").select("*").eq("Date", booking_date).execute()
        if not res.data: return True  # no slot record = assume available
        return (res.data[0].get("available") or 0) > 0
    except: return True

# ── Session manager ────────────────────────────────────────────────────────────
def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "step": "idle",
            "language": None,
            "collected": {},
            "history": [],
        }
    return sessions[session_id]

def reset_session(session_id: str):
    sessions[session_id] = {
        "step": "idle",
        "language": None,
        "collected": {},
        "history": [],
    }

# ── Language-aware responses ───────────────────────────────────────────────────
def r(english: str, roman: str, urdu: str, lang: str) -> str:
    if lang == 'urdu': return urdu
    if lang == 'roman_urdu': return roman
    return english

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

# ── Owner context builder ──────────────────────────────────────────────────────
def build_owner_context():
    try:
        bookings = supabase.table("bookings").select("*").order("Date", desc=True).limit(50).execute().data
        slots = supabase.table("slots").select("*").order("Date").limit(14).execute().data
        leads = supabase.table("leads").select("*").order("created_at", desc=True).limit(20).execute().data
        today = str(date.today())
        upcoming = [b for b in bookings if (b.get("Date") or "")[:10] >= today]
        unpaid = [b for b in bookings if (b.get("Payment Status") or "").lower() == "unpaid"]
        context = f"""
=== iRepair Shop — Live Data (as of {today}) ===

UPCOMING BOOKINGS ({len(upcoming)} total):
{chr(10).join([f"- {b.get('Date')} {b.get('Time')} | {b.get('Name')} | {b.get('Phone')} | {b.get('Device')} | {b.get('Service')} | {b.get('Status')} | {b.get('Payment Status')}" for b in upcoming[:20]]) or "None"}

UNPAID BOOKINGS ({len(unpaid)} total):
{chr(10).join([f"- {b.get('Date')} | {b.get('Name')} | {b.get('Phone')} | {b.get('Device')}" for b in unpaid[:10]]) or "None"}

SLOT AVAILABILITY:
{chr(10).join([f"- {s.get('Date')}: {s.get('available')} available, {s.get('booked')} booked" for s in slots]) or "No slot data"}

LEADS ({len(leads)} total, last 10):
{chr(10).join([f"- {l.get('Name')} | {l.get('Phone')} | {l.get('Device')} | {l.get('Issue')}" for l in leads[:10]]) or "None"}
""".strip()
        return context
    except Exception as e:
        return f"Error fetching data: {str(e)}"

# ── Owner chat ─────────────────────────────────────────────────────────────────
@router.post("/owner")
def owner_chat(req: OwnerChatRequest, user=Depends(verify_token)):
    try:
        if req.context:
            bookings = req.context.get("bookings", [])
            slots = req.context.get("slots", [])
            leads = req.context.get("leads", [])
            revenue = req.context.get("revenue", 0)
            today = str(date.today())
            context = f"""
=== iRepair Shop — Live Data (as of {today}) ===

BOOKINGS ({len(bookings)} total):
{chr(10).join([f"- {b.get('Date')} {b.get('Time')} | {b.get('Name')} | {b.get('Phone')} | {b.get('Device')} | {b.get('Service')} | {b.get('Status')} | {b.get('Payment Status')}" for b in bookings[:30]]) or "None"}

REVENUE (confirmed bookings): Rs{revenue:,}

SLOTS:
{chr(10).join([f"- {s.get('Date')}: {s.get('available')} available, {s.get('booked')} booked" for s in slots]) or "No slot data"}

LEADS ({len(leads)} total):
{chr(10).join([f"- {l.get('Name')} | {l.get('Phone')} | {l.get('Device')} | {l.get('Issue')}" for l in leads[:10]]) or "None"}
""".strip()
        else:
            context = build_owner_context()

        system_prompt = f"""You are iRepair Assistant — the smartest employee at an iPhone repair shop in Lahore called iRepair.

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
        res = groq_client.chat.completions.create(model=GROQ_MODEL, messages=messages, max_tokens=600, temperature=0.4)
        return {"reply": res.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Customer chat ──────────────────────────────────────────────────────────────
@router.post("/customer")
def customer_chat(req: CustomerChatRequest):
    try:
        session_id = req.session_id or str(uuid.uuid4())
        session = get_session(session_id)
        msg = req.message.strip()

        # ── Spam filter ──
        if is_spam(msg):
            return {
                "reply": "Please keep the conversation respectful. How can I help you book a repair? 😊",
                "session_id": session_id,
                "booking_created": False,
                "booking_info": None,
            }

        # ── Language lock ──
        if not session["language"]:
            session["language"] = detect_language(msg)
        lang = session["language"]

        # ── Exit intent ──
        if is_exit(msg) and session["step"] != "idle":
            save_lead(session["collected"])
            reset_session(session_id)
            reply = r(
                "No problem! I've saved your info and we'll reach out soon. Take care! 👋",
                "Koi baat nahi! Aapki info save kar li hai, hum jald contact karein ge. Allah Hafiz! 👋",
                "کوئی بات نہیں! آپ کی معلومات محفوظ کر لی ہیں۔ اللہ حافظ! 👋",
                lang
            )
            return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}

        step = session["step"]
        collected = session["collected"]

        # ── Add to history ──
        session["history"].append({"role": "user", "content": msg})

        # ════════════════════════════════════════════════
        # STEP: IDLE — classify intent
        # ════════════════════════════════════════════════
        if step == "idle":
            available_slots = get_available_slots()
            slots_text = "\n".join([f"- {s.get('Date')}: {s.get('available')} slots available" for s in available_slots]) or "Please call us to check availability"

            classify_prompt = f"""You are an intent classifier for a repair shop chatbot.
Classify the user message as either BOOKING or QUESTION.
BOOKING = user wants to book/schedule a repair appointment
QUESTION = user is asking about services, prices, location, hours, warranty, or anything else

Reply with ONLY one word: BOOKING or QUESTION

User message: {msg}"""

            classify_res = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": classify_prompt}],
                max_tokens=10, temperature=0,
            )
            intent = classify_res.choices[0].message.content.strip().upper()

            if "BOOKING" in intent:
                session["step"] = "get_device"
                reply = r(
                    f"Great! I'll help you book a repair. 🔧\n\nFirst, what device do you have and what's the issue? (e.g. iPhone 13, cracked screen)",
                    f"Zabardast! Main aapki booking mein madad karunga. 🔧\n\nPehle batayein, kaun sa device hai aur kya masla hai? (maslan: iPhone 13, screen toot gayi)",
                    f"بہت اچھا! میں آپ کی بکنگ میں مدد کروں گا۔ 🔧\n\nپہلے بتائیں، کون سا ڈیوائس ہے اور کیا مسئلہ ہے؟",
                    lang
                )
            else:
                # Answer business question via AI
                system_prompt = f"""You are a helpful assistant for iRepair — an iPhone repair shop in Lahore, Pakistan.

Answer the customer's question helpfully and concisely.
If you don't know the specific answer, say we offer professional repair services and they can ask more or book an appointment.
Reply in the same language as the customer (English, Roman Urdu, or Urdu).

SERVICES WE OFFER:
- Screen Repair
- Battery Replacement
- Water Damage Repair  
- Charging Port Repair
- Camera Repair
- Software Issues

AVAILABLE SLOTS:
{slots_text}

Keep response under 100 words. End with asking if they'd like to book an appointment."""

                history_msgs = [{"role": "system", "content": system_prompt}]
                history_msgs += session["history"][-6:]
                res = groq_client.chat.completions.create(model=GROQ_MODEL, messages=history_msgs, max_tokens=200, temperature=0.5)
                reply = res.choices[0].message.content

            session["history"].append({"role": "assistant", "content": reply})
            return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}

        # ════════════════════════════════════════════════
        # STEP: GET DEVICE + ISSUE
        # ════════════════════════════════════════════════
        elif step == "get_device":
            if len(msg) < 3:
                reply = r("Please tell me your device and issue (e.g. iPhone 14, battery draining fast)", "Device aur masla batayein (maslan: iPhone 14, battery jaldi khatam hoti hai)", "ڈیوائس اور مسئلہ بتائیں", lang)
                return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}

            collected["device_issue"] = msg
            # Try to split device from issue
            parts = msg.split(',', 1)
            collected["device"] = parts[0].strip()
            collected["issue"] = parts[1].strip() if len(parts) > 1 else msg

            session["step"] = "get_date"
            available_slots = get_available_slots()
            slots_text = "\n".join([f"• {s.get('Date')}" for s in available_slots[:7]]) or "Call us for availability"

            reply = r(
                f"Got it! **{collected['device']}** — {collected['issue']}.\n\nAvailable dates:\n{slots_text}\n\nWhich date works for you? (You can say 'tomorrow', 'Saturday', or a specific date)",
                f"Theek hai! **{collected['device']}** — {collected['issue']}.\n\nAvailable dates:\n{slots_text}\n\nKaun si date theek rahegi? (Kal, Saturday, ya koi specific date bata saktay hain)",
                f"ٹھیک ہے! دستیاب تاریخیں:\n{slots_text}\n\nکون سی تاریخ مناسب ہے؟",
                lang
            )
            session["history"].append({"role": "assistant", "content": reply})
            return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}

        # ════════════════════════════════════════════════
        # STEP: GET DATE
        # ════════════════════════════════════════════════
        elif step == "get_date":
            parsed = parse_date(msg)
            if not parsed:
                reply = r(
                    "I couldn't understand that date. Please say something like 'tomorrow', 'Saturday', or '2026-07-05'",
                    "Date samajh nahi aayi. 'Kal', 'Saturday', ya '2026-07-05' ki tarah batayein",
                    "تاریخ سمجھ نہیں آئی۔ کل، ہفتہ، یا مخصوص تاریخ بتائیں",
                    lang
                )
                return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}

            if parsed < str(date.today()):
                reply = r("That date has passed! Please choose a future date.", "Yeh date guzar gayi hai! Aagay ki date choose karein.", "یہ تاریخ گزر گئی! آگے کی تاریخ چنیں۔", lang)
                return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}

            if not is_slot_available(parsed):
                reply = r(f"Sorry, {parsed} is fully booked! Please choose another date.", f"Sorry, {parsed} ko slots full hain! Koi aur date choose karein.", f"معذرت، {parsed} کو سلاٹس بھرے ہوئے ہیں!", lang)
                return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}

            collected["date"] = parsed
            session["step"] = "get_time"
            reply = r(
                f"✅ {parsed} works!\n\nWhat time would you prefer? Our hours are 10 AM – 8 PM\n(e.g. 11 AM, 2 PM, 4:30 PM)",
                f"✅ {parsed} theek hai!\n\nKaun sa waqt chahiye? Hum 10 AM se 8 PM tak khule hain\n(maslan: 11 AM, 2 PM, 4:30 PM)",
                f"✅ {parsed} ٹھیک ہے!\n\nکیا وقت مناسب ہے؟ ہم 10 بجے سے 8 بجے تک کھلے ہیں",
                lang
            )
            session["history"].append({"role": "assistant", "content": reply})
            return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}

        # ════════════════════════════════════════════════
        # STEP: GET TIME
        # ════════════════════════════════════════════════
        elif step == "get_time":
            parsed_time = parse_time(msg)
            if not parsed_time:
                reply = r(
                    "Please give a valid time between 10 AM and 8 PM (e.g. '11 AM', '2:30 PM')",
                    "10 AM se 8 PM ke beech ka waqt dijiye (maslan: '11 AM', '2:30 PM')",
                    "10 بجے سے 8 بجے کے درمیان وقت دیں",
                    lang
                )
                return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}

            collected["time"] = parsed_time
            session["step"] = "get_name"
            reply = r("Perfect! Now, what's your full name?", "Zabardast! Aapka poora naam kya hai?", "بہت اچھا! آپ کا پورا نام کیا ہے؟", lang)
            session["history"].append({"role": "assistant", "content": reply})
            return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}

        # ════════════════════════════════════════════════
        # STEP: GET NAME
        # ════════════════════════════════════════════════
        elif step == "get_name":
            if len(msg.strip()) < 2 or re.match(r'^\d+$', msg.strip()):
                reply = r("Please enter a valid name.", "Sahi naam dijiye.", "درست نام درج کریں۔", lang)
                return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}

            collected["name"] = msg.strip().title()
            session["step"] = "get_phone"
            reply = r(
                "Got it! What's your phone number? (with country code, e.g. +923001234567 or 03001234567)",
                "Theek hai! Aapka phone number kya hai? (country code ke saath, maslan: +923001234567 ya 03001234567)",
                "آپ کا فون نمبر کیا ہے؟ (ملکی کوڈ کے ساتھ)",
                lang
            )
            session["history"].append({"role": "assistant", "content": reply})
            return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}

        # ════════════════════════════════════════════════
        # STEP: GET PHONE
        # ════════════════════════════════════════════════
        elif step == "get_phone":
            formatted = format_phone(msg)
            if not formatted:
                reply = r(
                    "That doesn't look like a valid phone number. Please enter with country code (e.g. +923001234567 or 03001234567)",
                    "Yeh phone number sahi nahi lagta. Country code ke saath dalein (maslan: 03001234567)",
                    "یہ فون نمبر درست نہیں لگتا۔ ملکی کوڈ کے ساتھ درج کریں",
                    lang
                )
                return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}

            # Duplicate booking check
            if has_duplicate_booking(formatted, collected.get("date", "")):
                reply = r(
                    f"You already have a booking on {collected.get('date')}! Would you like to choose a different date?",
                    f"Aapki {collected.get('date')} ko pehle se booking hai! Koi aur date choose karein ge?",
                    f"آپ کی {collected.get('date')} کو پہلے سے بکنگ ہے!",
                    lang
                )
                session["step"] = "get_date"
                return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}

            collected["phone"] = formatted
            session["step"] = "get_email"
            reply = r(
                "Great! What's your email address? (for booking confirmation)",
                "Theek hai! Aapka email address kya hai? (booking confirmation ke liye)",
                "آپ کا ای میل پتہ کیا ہے؟ (بکنگ کنفرمیشن کے لیے)",
                lang
            )
            session["history"].append({"role": "assistant", "content": reply})
            return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}

        # ════════════════════════════════════════════════
        # STEP: GET EMAIL
        # ════════════════════════════════════════════════
        elif step == "get_email":
            # Allow skipping email
            if any(w in msg.lower() for w in ['skip', 'no', 'nahi', 'nope', 'dont have', "don't have"]):
                collected["email"] = ""
                session["step"] = "confirm"
            elif not is_valid_email(msg):
                reply = r(
                    "That email doesn't look right. Please enter a valid email (e.g. name@gmail.com) or type 'skip'",
                    "Yeh email sahi nahi lagta. Sahi email dalein (maslan: name@gmail.com) ya 'skip' likhein",
                    "یہ ای میل درست نہیں۔ درست ای میل درج کریں یا 'skip' لکھیں",
                    lang
                )
                return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}
            else:
                collected["email"] = msg.strip().lower()
                session["step"] = "confirm"

            # Show confirmation summary
            reply = r(
                f"Please confirm your booking details:\n\n"
                f"📱 Device: {collected.get('device')}\n"
                f"🔧 Issue: {collected.get('issue')}\n"
                f"📅 Date: {collected.get('date')}\n"
                f"⏰ Time: {collected.get('time')}\n"
                f"👤 Name: {collected.get('name')}\n"
                f"📞 Phone: {collected.get('phone')}\n"
                f"{'📧 Email: ' + collected.get('email') if collected.get('email') else ''}\n\n"
                f"Type **YES** to confirm or **NO** to cancel.",

                f"Apni booking details confirm karein:\n\n"
                f"📱 Device: {collected.get('device')}\n"
                f"🔧 Masla: {collected.get('issue')}\n"
                f"📅 Date: {collected.get('date')}\n"
                f"⏰ Waqt: {collected.get('time')}\n"
                f"👤 Naam: {collected.get('name')}\n"
                f"📞 Phone: {collected.get('phone')}\n"
                f"{'📧 Email: ' + collected.get('email') if collected.get('email') else ''}\n\n"
                f"**YES** likhein confirm karne ke liye ya **NO** cancel karne ke liye.",

                f"اپنی بکنگ کی تفصیلات کنفرم کریں:\n\n"
                f"📱 ڈیوائس: {collected.get('device')}\n"
                f"🔧 مسئلہ: {collected.get('issue')}\n"
                f"📅 تاریخ: {collected.get('date')}\n"
                f"⏰ وقت: {collected.get('time')}\n"
                f"👤 نام: {collected.get('name')}\n"
                f"📞 فون: {collected.get('phone')}\n\n"
                f"تصدیق کے لیے **YES** لکھیں۔",
                lang
            )
            session["history"].append({"role": "assistant", "content": reply})
            return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}

        # ════════════════════════════════════════════════
        # STEP: CONFIRM
        # ════════════════════════════════════════════════
        elif step == "confirm":
            if msg.lower().strip() in ['yes', 'y', 'haan', 'ha', 'confirm', 'ok', 'okay', 'theek hai', 'theek', 'ji']:
                # Final slot check
                if not is_slot_available(collected.get("date", "")):
                    session["step"] = "get_date"
                    reply = r(
                        f"Sorry! That slot just got filled. Please choose another date.",
                        f"Sorry! Woh slot abhi full ho gaya. Koi aur date choose karein.",
                        f"معذرت! وہ سلاٹ ابھی بھر گیا۔ دوسری تاریخ چنیں۔",
                        lang
                    )
                    return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}

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
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Booking failed: {str(e)}")

                booking_info = {**collected, "booking_id": booking_id}
                reset_session(session_id)

                reply = r(
                    f"🎉 Booking confirmed! Your ID is **{booking_id}**.\n\nWe'll see you on {booking_info.get('date')} at {booking_info.get('time')}. Please arrive 5 minutes early. See you soon! 🔧",
                    f"🎉 Booking confirm ho gayi! Aapka ID hai **{booking_id}**.\n\nHum aapko {booking_info.get('date')} ko {booking_info.get('time')} baje milainge. 5 minute pehle aa jayein. Shukriya! 🔧",
                    f"🎉 بکنگ کنفرم ہو گئی! آپ کا ID ہے **{booking_id}**.\n\nہم آپ سے {booking_info.get('date')} کو ملیں گے۔ شکریہ! 🔧",
                    lang
                )
                return {"reply": reply, "session_id": session_id, "booking_created": True, "booking_info": booking_info}

            elif msg.lower().strip() in ['no', 'n', 'nahi', 'cancel', 'nope']:
                reset_session(session_id)
                reply = r("Booking cancelled. Feel free to start again anytime! 😊", "Booking cancel kar di. Jab chahein dobara shuru kar saktay hain! 😊", "بکنگ منسوخ کر دی۔ جب چاہیں دوبارہ شروع کریں! 😊", lang)
                return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}
            else:
                reply = r("Please type YES to confirm or NO to cancel.", "YES likhein confirm karne ke liye ya NO cancel ke liye.", "تصدیق کے لیے YES یا منسوخی کے لیے NO لکھیں۔", lang)
                return {"reply": reply, "session_id": session_id, "booking_created": False, "booking_info": None}

        # Fallback
        return {"reply": r("How can I help you?", "Main aapki kaise madad kar sakta hun?", "میں آپ کی کیسے مدد کر سکتا ہوں؟", lang), "session_id": session_id, "booking_created": False, "booking_info": None}

    except HTTPException:
        raise
    except Exception as e:
        return {"reply": "Sorry, something went wrong. Please try again.", "session_id": req.session_id or "", "booking_created": False, "booking_info": None}
