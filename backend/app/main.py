from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes_gestures

app = FastAPI(title="Touchpad Gesture Recognition API")

# السماح للواجهة بالتواصل (React/Electron)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # عدلها لاحقًا للأمان
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ربط المسارات
app.include_router(routes_gestures.router, prefix="/api/v1/gestures", tags=["Gestures"])

@app.get("/")
def read_root():
    return {"message": "API is running"}
