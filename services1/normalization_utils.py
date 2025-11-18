import requests
from typing import List, Dict, Any
import numpy as np

def normalize_positions(frames: List[Dict]) -> List[Dict]:
    """تطبيع الإحداثيات إلى [0,1] في كامل الجستشر"""
    xs = []
    ys = []
    for f in frames:
        for p in f["points"]:
            xs.append(p["x"])
            ys.append(p["y"])

    if not xs or not ys:
        return frames

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    w = max_x - min_x if max_x - min_x != 0 else 1
    h = max_y - min_y if max_y - min_y != 0 else 1

    for f in frames:
        for p in f["points"]:
            p["x"] = (p["x"] - min_x) / w
            p["y"] = (p["y"] - min_y) / h

    return frames


def resample_frames(frames: List[Dict], target_frames: int = 60) -> List[Dict]:
    """توحيد عدد الفريمات باستخدام interpolation"""
    if len(frames) == 0:
        return frames

    original = np.linspace(0, 1, len(frames))
    target = np.linspace(0, 1, target_frames)

    resampled = []
    for t in target:
        idx = np.searchsorted(original, t)
        idx = np.clip(idx, 1, len(frames) - 1)

        f1 = frames[idx - 1]
        f2 = frames[idx]

        alpha = (t - original[idx - 1]) / (original[idx] - original[idx - 1] + 1e-6)
        merged_points = f1["points"] if alpha < 0.5 else f2["points"]

        resampled.append({
            "timestamp": int((1 - alpha) * f1["timestamp"] + alpha * f2["timestamp"]),
            "delta_ms": int((1 - alpha) * f1["delta_ms"] + alpha * f2["delta_ms"]),
            "points": merged_points
        })

    return resampled


def resample_points_per_frame(frames: List[Dict], target_points: int = 20) -> List[Dict]:
    """توحيد عدد النقاط داخل كل فريم"""
    new_frames = []

    for f in frames:
        pts = f["points"]

        if len(pts) == 0:
            new_frames.append({**f, "points": [{"x": 0, "y": 0}] * target_points})
            continue

        xs = np.array([p["x"] for p in pts])
        ys = np.array([p["y"] for p in pts])

        original = np.linspace(0, 1, len(pts))
        target = np.linspace(0, 1, target_points)

        new_pts = []
        for t in target:
            idx = np.searchsorted(original, t)
            idx = np.clip(idx, 1, len(pts) - 1)

            x = xs[idx - 1] if t < original[idx] else xs[idx]
            y = ys[idx - 1] if t < original[idx] else ys[idx]

            new_pts.append({"x": float(x), "y": float(y)})

        new_frames.append({**f, "points": new_pts})

    return new_frames

