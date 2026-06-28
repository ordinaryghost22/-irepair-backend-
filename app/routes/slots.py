from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from supabase import create_client
from app.config import SUPABASE_URL, SUPABASE_KEY
from app.auth import verify_token

router = APIRouter()
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class SlotUpdate(BaseModel):
    available: Optional[int] = None
    booked: Optional[int] = None

@router.get("/")
def get_slots(user=Depends(verify_token)):
    try:
        res = supabase.table("slots").select("*").order("Date").execute()
        return res.data  # plain array
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{slot_id}")
def update_slot(slot_id: str, slot: SlotUpdate, user=Depends(verify_token)):
    try:
        data = {}
        if slot.available is not None: data["available"] = slot.available
        if slot.booked is not None: data["booked"] = slot.booked
        res = supabase.table("slots").update(data).eq("id", slot_id).execute()
        return res.data[0] if res.data else {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))