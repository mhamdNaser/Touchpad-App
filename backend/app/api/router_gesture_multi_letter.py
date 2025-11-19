
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.services.multi_char_prediction_pipeline import MultiCharPredictionPipeline
import numpy as np
from typing import Any


router = APIRouter()
multi_char_predictor = MultiCharPredictionPipeline(verbose=True)

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

def safe_convert(obj: Any) -> Any:
    """تحويل آمن لـ JSON"""
    if hasattr(obj, 'item'):  # إذا كان numpy type
        return obj.item()     # استخدم .item() لتحويله
    elif isinstance(obj, dict):
        return {k: safe_convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_convert(item) for item in obj]
    else:
        return obj


def normalize_gesture(frames):
    """تطبيع الإيماءة مثل التدريب"""
    if not frames:
        return frames
        
    # جمع جميع النقاط
    all_points = []
    for frame in frames:
        all_points.extend(frame["points"])
    
    if not all_points:
        return frames
        
    # حساب الصندوق المحيط
    xs = [p['x'] for p in all_points]
    ys = [p['y'] for p in all_points]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = max_x - min_x
    height = max_y - min_y
    
    if width == 0: width = 1
    if height == 0: height = 1
    
    # التطبيع
    scale = 2.0 / max(width, height)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    normalized_frames = []
    for frame in frames:
        normalized_points = []
        for point in frame['points']:
            normalized_points.append({
                'x': (point['x'] - center_x) * scale,
                'y': (point['y'] - center_y) * scale,
                'pressure': point.get('pressure', 1.0)
            })
        
        normalized_frames.append({
            'timestamp': frame.get('timestamp', frame.get('ts', 0)),
            'delta_ms': frame.get('delta_ms', 16),
            'points': normalized_points
        })
    
    return normalized_frames

@router.post("/predict")
def predict(gesture: GesturePayload):
    try:
        print(f"📥 Received gesture with {len(gesture.frames)} frames")
        
        frames = []
        for frame in gesture.frames:
            valid_points = [p.dict() for p in frame.points if not (p.x == 0 and p.y == 0)]
            if valid_points:
                frames.append({
                    "timestamp": frame.ts,
                    "delta_ms": frame.delta_ms,
                    "points": valid_points
                })

        if not frames:
            raise HTTPException(status_code=400, detail="No valid frames with points")

        frames = normalize_gesture(frames)

        gesture_dict = {
            "frames": frames,
            "duration_ms": gesture.duration_ms,
            "start_time": gesture.start_time,
            "end_time": gesture.end_time
        }

        result = multi_char_predictor.predict_gesture(gesture_dict)
        
        # 🔥 تحويل النتيجة باستخدام safe_convert
        return safe_convert(result)

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/analyze-gesture")
def analyze_gesture(gesture: GesturePayload):
    """تحليل مفصل للإيماءة"""
    try:
        frames = []
        for frame in gesture.frames:
            valid_points = [p.dict() for p in frame.points if not (p.x == 0 and p.y == 0)]
            if valid_points:
                frames.append({
                    "timestamp": frame.ts,
                    "delta_ms": frame.delta_ms,
                    "points": valid_points
                })

        gesture_dict = {
            "frames": frames,
            "duration_ms": gesture.duration_ms,
            "start_time": gesture.start_time,
            "end_time": gesture.end_time
        }

        analysis = multi_char_predictor.get_detailed_analysis(gesture_dict)
        return analysis

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@router.post("/predict-with-segmentation")
def predict_with_segmentation(gesture: GesturePayload, force_segmentation: bool = False):
    """التنبؤ مع خيار إجباري للتجزئة"""
    try:
        frames = []
        for frame in gesture.frames:
            valid_points = [p.dict() for p in frame.points if not (p.x == 0 and p.y == 0)]
            if valid_points:
                frames.append({
                    "timestamp": frame.ts,
                    "delta_ms": frame.delta_ms,
                    "points": valid_points
                })

        gesture_dict = {
            "frames": frames,
            "duration_ms": gesture.duration_ms,
            "start_time": gesture.start_time,
            "end_time": gesture.end_time
        }

        if force_segmentation and len(frames) > 15:
            # تجزئة إجبارية
            segments = multi_char_predictor.segmenter.detect_segments(frames)
            if len(segments) > 1:
                result = multi_char_predictor._predict_multi_char_gesture(gesture_dict, segments, {
                    "is_multi_char": True,
                    "segment_count": len(segments),
                    "confidence": 0.8,
                    "total_frames": len(frames)
                })
                return result

        # التنبؤ العادي
        result = multi_char_predictor.predict_gesture(gesture_dict)
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    





