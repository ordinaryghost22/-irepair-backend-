from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.auth import verify_password, hash_password, create_access_token
from app.config import OWNER_USERNAME, OWNER_PASSWORD

router = APIRouter()

class LoginRequest(BaseModel):
    username: str
    password: str

# Store hashed password
HASHED_PASSWORD = hash_password(OWNER_PASSWORD)

@router.post("/login")
def login(req: LoginRequest):
    if req.username != OWNER_USERNAME:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(req.password, HASHED_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"sub": req.username})
    return {
        "access_token": token,
        "token_type": "bearer",
        "message": "Login successful"
    }