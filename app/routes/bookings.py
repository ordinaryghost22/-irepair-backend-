from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from supabase import create_client
from app.config import SUPABASE_URL, SUPABASE_KEY
from app.auth import verify_token

router = APIRouter()
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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

class StatusUpdate(BaseModel):
    Status: str

class PaymentUpdate(BaseModel):
    payment_status: str

@router.get("/")
def get_bookings(user=Depends(verify_token)):
    try:
        res = supabase.table("bookings").select("*").order("Date", desc=True).execute()
        return res.data  # plain array
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{booking_id}")
def get_booking(booking_id: str, user=Depends(verify_token)):
    try:
        res = supabase.table("bookings").select("*").eq("Booking ID", booking_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Booking not found")
        return res.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/")
def create_booking(booking: Booking, user=Depends(verify_token)):
    try:
        data = {
            "Name": booking.name,
            "Phone": booking.phone,
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
        res = supabase.table("bookings").insert(data).execute()
        return res.data[0] if res.data else {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{booking_id}")
def update_booking(booking_id: str, booking: BookingUpdate, user=Depends(verify_token)):
    try:
        data = {}
        if booking.name is not None: data["Name"] = booking.name
        if booking.phone is not None: data["Phone"] = booking.phone
        if booking.email is not None: data["Email"] = booking.email
        if booking.device is not None: data["Device"] = booking.device
        if booking.service is not None: data["Service"] = booking.service
        if booking.issue is not None: data["Issue"] = booking.issue
        if booking.date is not None: data["Date"] = booking.date
        if booking.time is not None: data["Time"] = booking.time
        if booking.status is not None: data["Status"] = booking.status
        if booking.payment_status is not None: data["Payment Status"] = booking.payment_status
        if booking.notes is not None: data["Notes"] = booking.notes
        res = supabase.table("bookings").update(data).eq("Booking ID", booking_id).execute()
        return res.data[0] if res.data else {}
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