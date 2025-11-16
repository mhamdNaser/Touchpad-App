from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.services.prediction_pipeline import PredictionPipeline

router = APIRouter()
predictor = PredictionPipeline(max_timesteps=200)


class Point(BaseModel):
    x: float
    y: float
    pressure: float = 0.0


class Frame(BaseModel):
    ts: int
    frame_id: int
    points: List[Point]


class GesturePayload(BaseModel):
    start_time: int
    end_time: int
    duration_ms: int
    frame_count: int
    frames: List[Frame]


@router.post("/predict")
def predict(gesture: GesturePayload):
    try:
        # --- فلترة النقاط قبل المعالجة ---
        filtered_frames = []

        for frame in gesture.frames:
            # إبقاء النقاط الصحيحة فقط
            valid_points = [
                point for point in frame.points
                if not (point.x == 0 and point.y == 0)
            ]

            filtered_frames.append({
                "ts": frame.ts,
                "frame_id": frame.frame_id,
                "points": [p.dict() for p in valid_points]
            })

        # إرسال البيانات بعد الفلترة
        gesture_dict = {
            "start_time": gesture.start_time,
            "end_time": gesture.end_time,
            "duration_ms": gesture.duration_ms,
            "frame_count": len(filtered_frames),
            "frames": filtered_frames
        }

        result = predictor.predict_gesture(gesture_dict)
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
