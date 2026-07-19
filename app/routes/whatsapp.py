"""Manual WhatsApp Cloud API test helpers (JWT-protected)."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth import verify_token
from app.whatsapp import send_whatsapp_message

router = APIRouter()


class TestWhatsAppRequest(BaseModel):
    to: str = Field(..., description="Recipient phone (e.g. 03001234567 or +923001234567)")
    template_name: str = Field(
        default="hello_world",
        description="Approved template name (default: Meta hello_world)",
    )


@router.post("/test-whatsapp")
def test_whatsapp(body: TestWhatsAppRequest, user=Depends(verify_token)):
    """Send a template message to verify Cloud API credentials end-to-end."""
    result = send_whatsapp_message(body.to, body.template_name)
    if not result.get("ok"):
        status = result.get("status_code") or 400
        # Config / validation → 400; upstream Meta errors → 502
        if status >= 500 or result.get("status_code"):
            raise HTTPException(status_code=502, detail=result.get("error"))
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {
        "ok": True,
        "sent_by": user,
        "to": body.to,
        "template_name": body.template_name,
        "meta": result.get("data"),
    }
