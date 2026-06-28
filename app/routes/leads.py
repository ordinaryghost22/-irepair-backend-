from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from supabase import create_client
from app.config import SUPABASE_URL, SUPABASE_KEY
from app.auth import verify_token

router = APIRouter()
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class Lead(BaseModel):
    name: str
    phone: str
    device: Optional[str] = None
    issue: Optional[str] = None

@router.get("/")
def get_leads(user=Depends(verify_token)):
    try:
        res = supabase.table("leads").select("*").order("created_at", desc=True).execute()
        return res.data  # plain array
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/")
def create_lead(lead: Lead, user=Depends(verify_token)):
    try:
        data = {
            "Name": lead.name,
            "Phone": lead.phone,
            "Device": lead.device,
            "Issue": lead.issue,
        }
        res = supabase.table("leads").insert(data).execute()
        return res.data[0] if res.data else {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))