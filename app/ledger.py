"""
iRepair Verified Ledger
------------------------
Hash-chained, tamper-evident record of repairs and parts.
Each row's hash depends on its own data + the previous row's hash.
Editing any past row breaks every hash after it — instantly detectable.

Drop this file into your FastAPI backend (e.g. app/ledger.py).
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional
from supabase import Client

GENESIS_HASH = "0" * 64


def _compute_hash(data: dict, prev_hash: str) -> str:
    """Deterministic hash of a block's data + previous hash."""
    block_string = json.dumps({**data, "prev_hash": prev_hash}, sort_keys=True, default=str)
    return hashlib.sha256(block_string.encode()).hexdigest()


def _get_last_hash(supabase: Client, table: str) -> str:
    result = (
        supabase.table(table)
        .select("hash")
        .order("id", desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0]["hash"] if result.data else GENESIS_HASH


# ---------------------------------------------------------------------------
# Step 1 + 2 — Repair completion + warranty tracking
# ---------------------------------------------------------------------------

def log_repair(
    supabase: Client,
    booking_id: str,
    device_model: str,
    issue_type: str,
    technician_id: str,
    technician_name: Optional[str] = None,
    part_used: Optional[str] = None,
    part_serial_number: Optional[str] = None,
    warranty_days: int = 90,
) -> dict:
    """Call this the moment a technician marks a booking 'Completed'."""
    prev_hash = _get_last_hash(supabase, "repair_ledger")
    timestamp = datetime.utcnow()
    warranty_expiry = (timestamp + timedelta(days=warranty_days)).date()

    data = {
        "booking_id": str(booking_id),
        "device_model": device_model,
        "issue_type": issue_type,
        "part_used": part_used,
        "part_serial_number": part_serial_number,
        "technician_id": str(technician_id),
        "technician_name": technician_name,
        "warranty_expiry": str(warranty_expiry),
        "timestamp": timestamp.isoformat(),
    }
    block_hash = _compute_hash(data, prev_hash)
    row = {**data, "prev_hash": prev_hash, "hash": block_hash}

    result = supabase.table("repair_ledger").insert(row).execute()
    return result.data[0]


def get_repair(supabase: Client, booking_id: str) -> Optional[dict]:
    result = (
        supabase.table("repair_ledger")
        .select("*")
        .eq("booking_id", str(booking_id))
        .order("id", desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def check_warranty(supabase: Client, booking_id: str) -> dict:
    """Used by the chatbot to answer 'is my warranty still active?'"""
    record = get_repair(supabase, booking_id)
    if not record:
        return {"found": False}

    expiry = datetime.strptime(record["warranty_expiry"], "%Y-%m-%d").date()
    today = datetime.utcnow().date()
    return {
        "found": True,
        "active": expiry >= today,
        "warranty_expiry": record["warranty_expiry"],
        "device_model": record["device_model"],
    }


# ---------------------------------------------------------------------------
# Step 3 — Parts supplier verification
# ---------------------------------------------------------------------------

def log_part_received(
    supabase: Client,
    part_serial_number: str,
    supplier_name: str,
    part_name: Optional[str] = None,
) -> dict:
    """Call this when new parts arrive from a supplier, before any repair."""
    prev_hash = _get_last_hash(supabase, "parts_ledger")
    timestamp = datetime.utcnow()

    data = {
        "part_serial_number": part_serial_number,
        "supplier_name": supplier_name,
        "part_name": part_name,
        "received_date": timestamp.isoformat(),
    }
    block_hash = _compute_hash(data, prev_hash)
    row = {**data, "prev_hash": prev_hash, "hash": block_hash}

    result = supabase.table("parts_ledger").insert(row).execute()
    return result.data[0]


def trace_part(supabase: Client, part_serial_number: str) -> dict:
    """Full chain of custody: supplier -> repair (if used)."""
    part = (
        supabase.table("parts_ledger")
        .select("*")
        .eq("part_serial_number", part_serial_number)
        .execute()
    )
    repair = (
        supabase.table("repair_ledger")
        .select("*")
        .eq("part_serial_number", part_serial_number)
        .execute()
    )
    return {
        "part_received": part.data[0] if part.data else None,
        "used_in_repair": repair.data[0] if repair.data else None,
    }


# ---------------------------------------------------------------------------
# Step 4 — Technician accountability (pure read, no new logging)
# ---------------------------------------------------------------------------

def technician_stats(supabase: Client, technician_id: str) -> dict:
    repairs = (
        supabase.table("repair_ledger")
        .select("*")
        .eq("technician_id", str(technician_id))
        .execute()
        .data
    )
    return {
        "technician_id": technician_id,
        "verified_repairs": len(repairs),
        "chain_valid": verify_chain(supabase, "repair_ledger"),
    }


# ---------------------------------------------------------------------------
# Integrity check — run this on a schedule (e.g. daily cron) and expose
# it via an endpoint so anyone can confirm nothing was tampered with.
# ---------------------------------------------------------------------------

def verify_chain(supabase: Client, table: str) -> bool:
    rows = supabase.table(table).select("*").order("id").execute().data
    prev_hash = GENESIS_HASH

    for row in rows:
        data = {
            k: v for k, v in row.items()
            if k not in ("id", "prev_hash", "hash", "created_at")
        }
        expected_hash = _compute_hash(data, prev_hash)
        if expected_hash != row["hash"]:
            return False
        if row["prev_hash"] != prev_hash:
            return False
        prev_hash = row["hash"]

    return True
