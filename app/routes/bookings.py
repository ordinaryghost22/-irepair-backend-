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

@router.get("/")
def get_bookings(user=Depends(verify_token)):
    try:
        res = supabase.table("bookings").select("*").order("Date", desc=True).execute()
        return {"data": res.data, "count": len(res.data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{booking_id}")
def get_booking(booking_id: str, user=Depends(verify_token)):
    try:
        res = supabase.table("bookings").select("*").eq("id", booking_id).single().execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Booking not found")
        return res.data
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
        return {"message": "Booking created", "data": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{booking_id}")
def update_booking(booking_id: str, booking: BookingUpdate, user=Depends(verify_token)):
    try:
        data = {}
        if booking.name: data["Name"] = booking.name
        if booking.phone: data["Phone"] = booking.phone
        if booking.email: data["Email"] = booking.email
        if booking.device: data["Device"] = booking.device
        if booking.service: data["Service"] = booking.service
        if booking.issue: data["Issue"] = booking.issue
        if booking.date: data["Date"] = booking.date
        if booking.time: data["Time"] = booking.time
        if booking.status: data["Status"] = booking.status
        if booking.payment_status: data["Payment Status"] = booking.payment_status
        if booking.notes: data["Notes"] = booking.notes
        res = supabase.table("bookings").update(data).eq("id", booking_id).execute()
        return {"message": "Booking updated", "data": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{booking_id}")
def delete_booking(booking_id: str, user=Depends(verify_token)):
    try:
        supabase.table("bookings").delete().eq("id", booking_id).execute()
        return {"message": "Booking deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))