from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from app.routes import bookings, slots, leads, chat, auth, invoices, audit, cash_ledger, whatsapp

app = FastAPI(
    title="iRepair API",
    description="Backend for iRepair Dashboard",
    version="1.0.0"
)

# Trust Railway's proxy headers so HTTPS is correctly identified
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["Auth"])
app.include_router(bookings.router, prefix="/bookings", tags=["Bookings"])
app.include_router(slots.router, prefix="/slots", tags=["Slots"])
app.include_router(leads.router, prefix="/leads", tags=["Leads"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(invoices.router, prefix="/invoices", tags=["Invoices"])
app.include_router(audit.router, prefix="/audit-events", tags=["Audit"])
app.include_router(cash_ledger.router, prefix="/cash-ledger", tags=["Cash Ledger"])
app.include_router(whatsapp.router, tags=["WhatsApp"])

@app.get("/")
def root():
    return {"status": "ok", "message": "iRepair API is running"}