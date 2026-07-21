from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from supabase import create_client

from app.auth import verify_token
from app.config import SUPABASE_URL, SUPABASE_KEY
from app.ledger import (
    get_repair,
    check_warranty,
    log_part_received,
    trace_part,
    technician_stats,
    verify_chain,
)

router = APIRouter()
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


class PartReceivedPayload(BaseModel):
    part_serial_number: str
    supplier_name: str
    part_name: Optional[str] = None


@router.get("/verify/{booking_id}")
def verify_repair(booking_id: str):
    record = get_repair(supabase, booking_id)
    if not record:
        raise HTTPException(status_code=404, detail="No verified record found for this booking")

    chain_valid = verify_chain(supabase, "repair_ledger")
    return {
        "verified": chain_valid,
        "device_model": record["device_model"],
        "issue_type": record["issue_type"],
        "part_used": record["part_used"],
        "technician_name": record["technician_name"],
        "warranty_expiry": record["warranty_expiry"],
        "timestamp": record["timestamp"],
        "hash": record["hash"],
    }


@router.get("/warranty/{booking_id}")
def warranty_status(booking_id: str):
    return check_warranty(supabase, booking_id)


@router.post("/parts/received")
def receive_part(payload: PartReceivedPayload, user=Depends(verify_token)):
    entry = log_part_received(
        supabase,
        part_serial_number=payload.part_serial_number,
        supplier_name=payload.supplier_name,
        part_name=payload.part_name,
    )
    return {"logged": True, "hash": entry["hash"]}


@router.get("/parts/{part_serial_number}/trace")
def part_trace(part_serial_number: str, user=Depends(verify_token)):
    return trace_part(supabase, part_serial_number)


@router.get("/technicians/{technician_id}/stats")
def tech_stats(technician_id: str, user=Depends(verify_token)):
    return technician_stats(supabase, technician_id)


@router.get("/ledger/integrity")
def ledger_integrity(user=Depends(verify_token)):
    return {
        "repair_ledger_valid": verify_chain(supabase, "repair_ledger"),
        "parts_ledger_valid": verify_chain(supabase, "parts_ledger"),
    }
