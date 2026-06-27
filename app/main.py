from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import bookings, slots, leads, chat, auth

app = FastAPI(
    title="iRepair API",
    description="Backend for iRepair Dashboard",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://irepair-dashboard.vercel.app", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["Auth"])
app.include_router(bookings.router, prefix="/bookings", tags=["Bookings"])
app.include_router(slots.router, prefix="/slots", tags=["Slots"])
app.include_router(leads.router, prefix="/leads", tags=["Leads"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])

@app.get("/")
def root():
    return {"status": "ok", "message": "iRepair API is running"}