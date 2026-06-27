from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from groq import Groq
from supabase import create_client
from app.config import SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY
from app.auth import verify_token

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
    session_id: Optional[str] = None

def build_owner_context():
    try:
        bookings = supabase.table("bookings").select("*").order("Date", desc=True).limit(50).execute().data
        slots = supabase.table("slots").select("*").order("date").limit(14).execute().data
        leads = supabase.table("leads").select("*").order("created_at", desc=True).limit(20).execute().data

        from datetime import date
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
{chr(10).join([f"- {s.get('date')}: {s.get('available')} available, {s.get('booked')} booked" for s in slots]) or "No slot data"}

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
        slots = supabase.table("slots").select("*").order("date").limit(7).execute().data
        available_slots = [s for s in slots if (s.get("available") or 0) > 0]

        system_prompt = f"""You are a friendly booking assistant for iRepair — an iPhone repair shop in Lahore, Pakistan.

Your job:
- Help customers book appointments
- Answer questions about services and availability
- Collect: name, phone, device, issue, preferred date and time
- Be friendly, helpful, and concise
- Reply in the same language the customer uses (Urdu/Roman Urdu/English)

AVAILABLE SLOTS:
{chr(10).join([f"- {s.get('date')}: {s.get('available')} slots available" for s in available_slots]) or "Please call us for availability"}

SERVICES: Screen Repair, Battery Replacement, Water Damage, Charging Port, Camera Repair, Software Issues

When customer provides all details (name, phone, device, issue, date, time), confirm the booking and say the shop will contact them shortly."""

        res = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.message}
            ],
            max_tokens=400,
            temperature=0.5,
        )
        reply = res.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))