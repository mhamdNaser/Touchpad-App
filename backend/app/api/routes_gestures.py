from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.services.prediction_pipeline import PredictionPipeline
import numpy as np

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


def normalize_positions(frames):
    xs, ys = [], []
    for f in frames:
        for p in f["points"]:
            xs.append(p["x"])
            ys.append(p["y"])
    if not xs or not ys: return frames
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w, h = max(max_x - min_x, 1), max(max_y - min_y, 1)
    for f in frames:
        for p in f["points"]:
            p["x"] = (p["x"] - min_x) / w
            p["y"] = (p["y"] - min_y) / h
    return frames


def resample_frames(frames, target_frames):
    if len(frames) == 0: return frames
    original = np.linspace(0, 1, len(frames))
    target = np.linspace(0, 1, target_frames)
    resampled = []
    for t in target:
        idx = np.searchsorted(original, t)
        idx = np.clip(idx, 1, len(frames)-1)
        f1, f2 = frames[idx-1], frames[idx]
        alpha = (t - original[idx-1]) / (original[idx] - original[idx-1] + 1e-6)
        merged_points = f1["points"] if alpha < 0.5 else f2["points"]
        resampled.append({
            "timestamp": int((1-alpha)*f1["timestamp"] + alpha*f2["timestamp"]),
            "delta_ms": int((1-alpha)*f1["delta_ms"] + alpha*f2["delta_ms"]),
            "points": merged_points
        })
    return resampled


def resample_points_per_frame(frames, target_points):
    new_frames = []
    for f in frames:
        pts = f["points"]
        if len(pts) == 0:
            new_frames.append({**f, "points": [{"x":0,"y":0}]*target_points})
            continue
        xs = np.array([p["x"] for p in pts])
        ys = np.array([p["y"] for p in pts])
        original = np.linspace(0, 1, len(pts))
        target = np.linspace(0, 1, target_points)
        new_pts = []
        for t in target:
            idx = np.searchsorted(original, t)
            idx = np.clip(idx, 1, len(pts)-1)
            x = xs[idx-1] if t < original[idx] else xs[idx]
            y = ys[idx-1] if t < original[idx] else ys[idx]
            new_pts.append({"x": float(x), "y": float(y)})
        new_frames.append({**f, "points": new_pts})
    return new_frames


@router.post("/predict")
def predict(gesture: GesturePayload):
    try:
        # --- تحويل البيانات الخام ل dict ---
        frames = []
        for frame in gesture.frames:
            valid_points = [p.dict() for p in frame.points if not (p.x == 0 and p.y == 0)]
            frames.append({
                "timestamp": frame.ts,
                "delta_ms": 0,  # ممكن حسب الحاجة
                "points": valid_points
            })

        # --- preprocessing مثل التدريب ---
        frames = normalize_positions(frames)
        frames = resample_frames(frames, target_frames=predictor.max_timesteps)
        frames = resample_points_per_frame(frames, target_points=20)  # نفس العدد اللي استخدمته أثناء التدريب

        gesture_dict = {
            "frames": frames,
            "frame_count": len(frames),
            "duration_ms": gesture.duration_ms,
            "start_time": gesture.start_time,
            "end_time": gesture.end_time
        }

        result = predictor.predict_gesture(gesture_dict)
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
