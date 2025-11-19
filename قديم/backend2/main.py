# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes_gestures import router as gestures_router
import logging

app = FastAPI(title="Arabic Gesture Recognition API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(gestures_router, prefix="/api/gestures", tags=["Gestures"])

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@app.get("/")
def root():
    return {"message": "Arabic Gesture Recognition API running"}

@app.get("/health")
def health():
    return {"status": "healthy"}