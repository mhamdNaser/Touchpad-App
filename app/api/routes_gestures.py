from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from app.services.prediction_pipeline import PredictionPipeline
import numpy as np

router = APIRouter()
predictor = PredictionPipeline(verbose=True)  # بدون max_timesteps

class Point(BaseModel):
    x: float
    y: float
    pressure: float = 1.0

class Frame(BaseModel):
    ts: int
    frame_id: int
    points: List[Point]
    delta_ms: int = 16

class GesturePayload(BaseModel):
    start_time: int
    end_time: int
    duration_ms: int
    frame_count: int
    frames: List[Frame]


class FeedbackRequest(BaseModel):
    gesture_data: Dict
    predicted_char: str
    correct_char: str
    user_id: str = "anonymous"
    session_id: Optional[str] = None

class TrainingDataRequest(BaseModel):
    password: str  # لحماية endpoint

# ------------------ Utilities ------------------

def normalize_gesture(frames):
    """تطبيع الإيماءة مثل التدريب"""
    if not frames:
        return []

    all_points = [p for frame in frames for p in frame["points"]]
    if not all_points:
        return []

    xs = [p['x'] for p in all_points]
    ys = [p['y'] for p in all_points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max_x - min_x or 1
    height = max_y - min_y or 1
    scale = 2.0 / max(width, height)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    normalized_frames = []
    for frame in frames:
        normalized_points = [
            {
                'x': (p['x'] - center_x) * scale,
                'y': (p['y'] - center_y) * scale,
                'pressure': p.get('pressure', 1.0)
            }
            for p in frame['points']
        ]
        normalized_frames.append({
            'timestamp': frame.get('ts', 0),
            'delta_ms': frame.get('delta_ms', 16),
            'points': normalized_points
        })
    return normalized_frames

# ------------------ API Endpoints ------------------

@router.post("/predict")
def predict(gesture: GesturePayload):
    try:
        print(f"📥 Received gesture with {len(gesture.frames)} frames")

        frames = []
        for frame in gesture.frames:
            valid_points = [p.dict() for p in frame.points if not (p.x == 0 and p.y == 0)]
            if valid_points:
                frames.append({
                    "ts": frame.ts,
                    "delta_ms": frame.delta_ms,
                    "points": valid_points
                })

        if not frames:
            raise HTTPException(status_code=400, detail="No valid points in gesture")

        frames = normalize_gesture(frames)

        gesture_dict = {
            "frames": frames,
            "duration_ms": gesture.duration_ms,
            "start_time": gesture.start_time,
            "end_time": gesture.end_time
        }

        # استخدام الدالة الجديدة Top3
        result = predictor.predict_gesture_top3(gesture_dict)

        return {
            "predicted_char": result["predicted_char"],
            "confidence": result["confidence"],
            "top3": result["top3"]
        }

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))