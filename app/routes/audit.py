from fastapi import APIRouter, HTTPException, Depends, Query
from supabase import create_client
from app.config import SUPABASE_URL, SUPABASE_KEY
from app.auth import verify_token

router = APIRouter()
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


@router.get("/")
def list_audit_events(
    user=Depends(verify_token),
    limit: int = Query(200, ge=1, le=500),
):
    """Return recent audit events, newest first."""
    try:
        res = (
            supabase.table("audit_events")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
