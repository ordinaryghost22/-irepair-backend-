from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import uuid
from supabase import create_client
from app.config import SUPABASE_URL, SUPABASE_KEY
from app.auth import verify_token
from app.phone import normalize_phone
from app.slot_claim import claim_slot, link_slot_booking, release_slot, SLOT_UNAVAILABLE_MSG
from app.routes.reminders import send_booking_confirmation, schedule_reminder

router = APIRouter()
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# your business name - hardcode for now, or pull from a settings table later
BUSINESS_NAME = "iRepair"


class Booking(BaseModel):
    name: str
    phone: str
    email: Optional[str] = None
    device: Optional[str] = None
    service: Optional[str] = None
    issue: Optional[str] = None
    date: str
    time: str
    status: Optional[str] = "Pending"
    payment_status: Optional[str] = "Unpaid"
    notes: Optional[str] = None
    amount: Optional[float] = None
    source: Optional[str] = None

class BookingUpdate(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    device: Optional[str] = None
    service: Optional[str] = None
    issue: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    status: Optional[str] = None
    payment_status: Optional[str] = None
    notes: Optional[str] = None
    amount: Optional[float] = None
    source: Optional[str] = None

class StatusUpdate(BaseModel):
    Status: str

class PaymentUpdate(BaseModel):
    payment_status: str


def parse_appointment_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    """
    Tries a handful of common date/time format combos so we don't
    need to know the exact frontend format in advance.
    Add more formats to this list if none of these match.
    """
    date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"]
    time_formats = ["%H:%M", "%I:%M %p", "%I:%M%p"]

    for d_fmt in date_formats:
        for t_fmt in time_formats:
            try:
                return datetime.strptime(f"{date_str} {time_str}", f"{d_fmt} {t_fmt}")
            except ValueError:
                continue

    print(f"Could not parse date/time: '{date_str}' '{time_str}' — skipping email reminder")
    return None


@router.get("/")
def get_bookings(user=Depends(verify_token)):
    try:
        res = supabase.table("bookings").select("*").order("Date", desc=True).execute()
        return res.data  # plain array
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{booking_id}/history")
def get_booking_history(booking_id: str, user=Depends(verify_token)):
    """Return the chatbot conversation history linked to this booking (read-only)."""
    try:
        res = (
            supabase.table("chat_sessions")
            .select("history, updated_at, session_id")
            .eq("booking_id", booking_id)
            .limit(1)
            .execute()
        )
        if not res.data:
            return []
        history = res.data[0].get("history") or []
        # history is already chronological; no per-message timestamps stored
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{booking_id}")
def get_booking(booking_id: str, user=Depends(verify_token)):
    try:
        res = supabase.table("bookings").select("*").eq("Booking ID", booking_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Booking not found")
        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/")
def create_booking(booking: Booking):
    try:
        phone = normalize_phone(booking.phone)
        if not phone:
            raise HTTPException(
                status_code=400,
                detail="Invalid phone number. Use format 03001234567 or +923001234567",
            )
        # Same duplicate guard as chatbot: one booking per phone+date
        try:
            dup = (
                supabase.table("bookings")
                .select("Booking ID")
                .eq("Phone", phone)
                .eq("Date", booking.date)
                .execute()
            )
            if dup.data:
                raise HTTPException(
                    status_code=409,
                    detail=f"A booking already exists for this phone on {booking.date}",
                )
        except HTTPException:
            raise
        except Exception as dup_err:
            print(f"Duplicate check failed (continuing): {dup_err}")

        # Claim slot FIRST (atomic WHERE Status=Available). Reject if 0 rows.
        claimed = claim_slot(booking.date, booking.time, phone)
        if not claimed:
            raise HTTPException(status_code=409, detail=SLOT_UNAVAILABLE_MSG)

        slot_id = claimed.get("id")
        booking_id = f"CUST-{uuid.uuid4().hex[:8].upper()}"
        data = {
            "Booking ID": booking_id,
            "Name": booking.name,
            "Phone": phone,
            "Email": booking.email,
            "Device": booking.device,
            "Service": booking.service,
            "Issue": booking.issue,
            "Date": booking.date,
            "Time": booking.time,
            "Status": booking.status,
            "Payment Status": booking.payment_status,
            "Notes": booking.notes,
        }
        if booking.amount is not None:
            data["amount"] = booking.amount
        if booking.source:
            data["Source"] = booking.source

        try:
            res = supabase.table("bookings").insert(data).execute()
            result = res.data[0] if res.data else {}
            if slot_id is not None:
                try:
                    link_slot_booking(slot_id, booking_id)
                except Exception as link_err:
                    print(f"Slot Booking ID link failed (booking still saved): {link_err}")
        except Exception:
            if slot_id is not None:
                release_slot(slot_id)
            raise

        # --- NEW: send confirmation + schedule reminder ---
        if booking.email:
            try:
                appointment_dt = parse_appointment_datetime(booking.date, booking.time)
                if appointment_dt:
                    send_booking_confirmation(
                        customer_email=booking.email,
                        business_name=BUSINESS_NAME,
                        service=booking.service or "repair",
                        appointment_time=appointment_dt,
                    )
                    schedule_reminder(
                        customer_email=booking.email,
                        business_name=BUSINESS_NAME,
                        service=booking.service or "repair",
                        appointment_time=appointment_dt,
                    )
            except Exception as email_err:
                # don't let email failures break the booking itself
                print(f"Confirmation/reminder email failed: {email_err}")
        # --- END NEW ---

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{booking_id}")
def update_booking(booking_id: str, booking: BookingUpdate, user=Depends(verify_token)):
    try:
        data = {}
        if booking.name is not None: data["Name"] = booking.name
        if booking.phone is not None:
            phone = normalize_phone(booking.phone)
            if not phone:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid phone number. Use format 03001234567 or +923001234567",
                )
            data["Phone"] = phone
        if booking.email is not None: data["Email"] = booking.email
        if booking.device is not None: data["Device"] = booking.device
        if booking.service is not None: data["Service"] = booking.service
        if booking.issue is not None: data["Issue"] = booking.issue
        if booking.date is not None: data["Date"] = booking.date
        if booking.time is not None: data["Time"] = booking.time
        if booking.status is not None: data["Status"] = booking.status
        if booking.payment_status is not None: data["Payment Status"] = booking.payment_status
        if booking.notes is not None: data["Notes"] = booking.notes
        if booking.amount is not None: data["amount"] = booking.amount
        if booking.source is not None: data["Source"] = booking.source
        res = supabase.table("bookings").update(data).eq("Booking ID", booking_id).execute()
        return res.data[0] if res.data else {}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{booking_id}/status")
def update_booking_status(booking_id: str, body: StatusUpdate, user=Depends(verify_token)):
    try:
        res = supabase.table("bookings").update({"Status": body.Status}).eq("Booking ID", booking_id).execute()
        return res.data[0] if res.data else {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{booking_id}/payment")
def update_booking_payment(booking_id: str, body: PaymentUpdate, user=Depends(verify_token)):
    try:
        res = supabase.table("bookings").update({"Payment Status": body.payment_status}).eq("Booking ID", booking_id).execute()
        return res.data[0] if res.data else {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{booking_id}")
def delete_booking(booking_id: str, user=Depends(verify_token)):
    try:
        supabase.table("bookings").delete().eq("Booking ID", booking_id).execute()
        return {"message": "Booking deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))