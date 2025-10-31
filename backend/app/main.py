from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from app.core.database import SessionLocal, engine
from app.models import gesture, frame, point

# إنشاء الجداول في حال غير موجودة (اختياري)
gesture.Base.metadata.create_all(bind=engine)

app = FastAPI()

# دالة تعتمد على Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/gestures")
def get_gestures(db: Session = Depends(get_db)):
    gestures = db.query(gesture.Gesture).limit(10).all()
    return [
        {"id": g.id, "character": g.character, "duration_ms": g.duration_ms}
        for g in gestures
    ]
