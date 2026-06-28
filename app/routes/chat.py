from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from groq import Groq
from supabase import create_client
from app.config import SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY
from app.auth import verify_token
import uuid
from datetime import date

router = APIRouter()
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

GROQ_MODEL = "llama-3.3-70b-versatile"

class Message(BaseModel):
    role: str
    content: str

class OwnerChatRequest(BaseModel):
    messages: List[Message]

class CustomerChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []

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

@router.post("/owner")
def owner_chat(req: OwnerChatRequest, user=Depends(verify_token)):
    try:
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

        res = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=600,
            temperature=0.4,
        )
        reply = res.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/customer")
def customer_chat(req: CustomerChatRequest):
    try:
        f"- {s.get('Date')}: {s.get('available')} slots available"
        available_slots = [s for s in slots if (s.get("available") or 0) > 0]

        system_prompt = f"""You are a friendly booking assistant for iRepair — an iPhone repair shop in Lahore, Pakistan.

YOUR JOB:
- Help customers book repair appointments
- Collect info step by step: name, phone number, device, issue, preferred date, preferred time
- Be friendly, concise, and helpful
- Reply in the same language the customer uses (English, Urdu, or Roman Urdu)

AVAILABLE SLOTS:
{chr(10).join([f"- {s.get('date')}: {s.get('available')} slots available" for s in available_slots]) or "Please call us to check availability"}

SERVICES WE OFFER:
- Screen Repair
- Battery Replacement  
- Water Damage Repair
- Charging Port Repair
- Camera Repair
- Software Issues

BOOKING FLOW:
1. Greet and ask what device and issue they have
2. Ask for their name and phone number
3. Ask for preferred date and time (from available slots above)
4. Once you have ALL of: name, phone, device, issue, date, time — output EXACTLY this on its own line:
BOOK:name=<name>|phone=<phone>|device=<device>|issue=<issue>|date=<date>|time=<time>
5. Then confirm to the customer their booking is confirmed

IMPORTANT: Only output the BOOK: line when you have ALL 6 pieces of info."""

        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": m.role, "content": m.content} for m in (req.history or [])]
        messages.append({"role": "user", "content": req.message})

        res = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=400,
            temperature=0.5,
        )
        reply = res.choices[0].message.content

        # Parse and auto-create booking if AI collected all info
        booking_created = False
        if "BOOK:" in reply:
            try:
                book_line = [l for l in reply.split("\n") if l.startswith("BOOK:")][0]
                parts = book_line.replace("BOOK:", "").strip().split("|")
                info = {}
                for p in parts:
                    k, v = p.split("=", 1)
                    info[k.strip()] = v.strip()

                booking_id = f"CUST-{uuid.uuid4().hex[:8].upper()}"
                supabase.table("bookings").insert({
                    "Booking ID": booking_id,
                    "Name": info.get("name", ""),
                    "Phone": info.get("phone", ""),
                    "Device": info.get("device", ""),
                    "Issue": info.get("issue", ""),
                    "Service": info.get("issue", ""),
                    "Date": info.get("date", ""),
                    "Time": info.get("time", ""),
                    "Status": "Pending",
                    "Payment Status": "Unpaid",
                    "Notes": "Booked via customer chatbot",
                }).execute()
                booking_created = True
                reply = reply.replace(book_line, "").strip()
            except Exception as e:
                print(f"Booking creation error: {e}")

        return {"reply": reply, "booking_created": booking_created}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))