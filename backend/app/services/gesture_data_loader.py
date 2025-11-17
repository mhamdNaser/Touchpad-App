import requests
from typing import List, Dict, Any
import numpy as np
from app.services.normalization_utils import (
    normalize_positions,
    resample_frames,
    resample_points_per_frame
)

class GestureDataLoader:
    def __init__(self, api_url: str = "https://api.sydev.site/api/gestures", per_page: int = 50,
                 target_frames: int = 60, target_points: int = 20):
        self.api_url = api_url
        self.per_page = per_page
        self.target_frames = target_frames
        self.target_points = target_points

        self.session = requests.Session()
        self.session.timeout = 30
        self.allowed_states = ("down", "move")

    def _frame_sort_key(self, frame: Dict[str, Any]):
        return frame.get("timestamp") or frame.get("ts") or frame.get("frame_id") or 0

    def _process_gesture(self, gesture: Dict[str, Any]) -> Dict:
        """→ Normalize + Resample"""
        try:
            frames_raw = gesture.get("frames", [])
            points_raw = gesture.get("points", [])

            frames: List[Dict] = []

            # --------------------------
            # 1) إعادة بناء الفريمات الخام
            # --------------------------
            if frames_raw:
                sorted_frames = sorted(frames_raw, key=self._frame_sort_key)
                prev_ts = None
                for f in sorted_frames:
                    ts = f.get("timestamp") or f.get("ts") or 0
                    delta = f.get("delta_ms", 0)
                    if not delta and prev_ts is not None:
                        delta = ts - prev_ts
                    prev_ts = ts

                    pts = f.get("points", [])
                    clean_pts = [
                        {"x": p.get("x", 0.0), "y": p.get("y", 0.0)}
                        for p in pts if p.get("state") in self.allowed_states
                    ]

                    frames.append({
                        "timestamp": ts,
                        "delta_ms": delta,
                        "points": clean_pts
                    })

            elif points_raw:
                temp = {}
                for p in points_raw:
                    fid = p.get("frame_id")
                    if fid not in temp:
                        temp[fid] = {
                            "timestamp": p.get("timestamp", 0),
                            "delta_ms": p.get("delta_ms", 0),
                            "points": []
                        }
                    if p.get("state") in self.allowed_states:
                        temp[fid]["points"].append({"x": p["x"], "y": p["y"]})

                frames = sorted(list(temp.values()), key=self._frame_sort_key)
            else:
                return None

            # ========== STAGE 1: NORMALIZE ==========
            frames = normalize_positions(frames)

            # ========== STAGE 2: RESAMPLE FRAMES ==========
            frames = resample_frames(frames, self.target_frames)

            # ========== STAGE 3: RESAMPLE POINTS ==========
            frames = resample_points_per_frame(frames, self.target_points)

            return {
                "gesture_id": gesture.get("id"),
                "character": gesture.get("character"),
                "duration_ms": gesture.get("duration_ms", 0),
                "frames": frames,
                "frame_count": len(frames)
            }

        except Exception:
            return None

    def load_all_gestures(self) -> List[Dict]:
        """تحميل البيانات + Normalize + Resample"""
        page = 1
        all_gestures = []

        while True:
            url = f"{self.api_url}?page={page}&per_page={self.per_page}"
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            gestures = data.get("data", [])

            if not gestures:
                break

            for g in gestures:
                processed = self._process_gesture(g)
                if processed:
                    all_gestures.append(processed)

            page += 1

        return all_gestures