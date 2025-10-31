from fastapi import APIRouter
from app.schemas.gesture import GestureCreate
from app.services.gesture_recognition import recognize_gesture

router = APIRouter()

@router.post("/recognize")
def recognize_gesture_api(gesture: GestureCreate):
    result = recognize_gesture(gesture.data)
    return {"recognized_character": result}
