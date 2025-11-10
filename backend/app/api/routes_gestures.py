from fastapi import APIRouter
from app.schemas.gesture import GestureData
from app.services.predict_model import predict_gesture

router = APIRouter()

@router.post("/predict")
async def predict_gesture_route(gesture: GestureData):
    return predict_gesture(gesture)
