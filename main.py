from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes_gestures

app = FastAPI(title="Arabic Gesture Recognition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_gestures.router, prefix="/api/gestures", tags=["Gestures"])

@app.get("/")
def root():
    return {"message": "Gesture Recognition API running"}
