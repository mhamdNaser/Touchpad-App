# app/services/gesture_data_loader.py
from typing import List, Dict
import numpy as np
import requests

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

    def _frame_sort_key(self, frame: Dict):
        return frame.get("timestamp") or frame.get("ts") or frame.get("frame_id") or 0

    def normalize_positions(self, frames: List[Dict]) -> List[Dict]:
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

    def resample_frames(self, frames: List[Dict], target_frames: int = None) -> List[Dict]:
        target_frames = target_frames or self.target_frames
        if len(frames) == 0:
            return frames

        original = np.linspace(0, 1, len(frames))
        target = np.linspace(0, 1, target_frames)
        resampled = []

        # عدد النقاط الأكبر مؤقتًا لكل إطار لضمان interpolation سلسة
        max_points = max(len(f["points"]) for f in frames)
        temp_frames = self.resample_points_per_frame(frames, max_points)

        for t in target:
            idx = np.searchsorted(original, t)
            idx = np.clip(idx, 1, len(frames)-1)
            f1, f2 = temp_frames[idx-1], temp_frames[idx]
            alpha = (t - original[idx-1]) / (original[idx] - original[idx-1] + 1e-6)

            # interpolation حقيقية لكل نقطة
            interpolated_points = []
            for p1, p2 in zip(f1["points"], f2["points"]):
                x = (1-alpha)*p1["x"] + alpha*p2["x"]
                y = (1-alpha)*p1["y"] + alpha*p2["y"]
                interpolated_points.append({"x": x, "y": y})

            resampled.append({
                "timestamp": int((1-alpha)*f1["timestamp"] + alpha*f2["timestamp"]),
                "delta_ms": int((1-alpha)*f1["delta_ms"] + alpha*f2["delta_ms"]),
                "points": interpolated_points
            })

        return resampled


    def resample_points_per_frame(self, frames: List[Dict], target_points: int = None) -> List[Dict]:
        target_points = target_points or self.target_points
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
                alpha = (t - original[idx-1]) / (original[idx] - original[idx-1] + 1e-6)
                x = (1-alpha)*xs[idx-1] + alpha*xs[idx]
                y = (1-alpha)*ys[idx-1] + alpha*ys[idx]
                new_pts.append({"x": float(x), "y": float(y)})

            new_frames.append({**f, "points": new_pts})

        return new_frames

    def _process_gesture(self, gesture: Dict) -> Dict:
        frames_raw = gesture.get("frames", [])
        points_raw = gesture.get("points", [])
        frames = []
        if frames_raw:
            sorted_frames = sorted(frames_raw, key=self._frame_sort_key)
            prev_ts = None
            for f in sorted_frames:
                ts = f.get("timestamp") or f.get("ts") or 0
                delta = f.get("delta_ms", 0)
                if not delta and prev_ts is not None:
                    delta = ts - prev_ts
                prev_ts = ts
                pts = [{"x": p.get("x",0.0),"y":p.get("y",0.0)}
                       for p in f.get("points",[]) if p.get("state") in self.allowed_states]
                frames.append({"timestamp":ts,"delta_ms":delta,"points":pts})
        elif points_raw:
            temp = {}
            for p in points_raw:
                fid = p.get("frame_id")
                if fid not in temp:
                    temp[fid] = {"timestamp":p.get("timestamp",0),"delta_ms":p.get("delta_ms",0),"points":[]}
                if p.get("state") in self.allowed_states:
                    temp[fid]["points"].append({"x":p["x"],"y":p["y"]})
            frames = sorted(list(temp.values()), key=self._frame_sort_key)
        else:
            return None
        frames = self.normalize_positions(frames)
        frames = self.resample_frames(frames, self.target_frames)
        frames = self.resample_points_per_frame(frames, self.target_points)
        for f in frames:
            prev_x, prev_y = None, None
            for p in f["points"]:
                if prev_x is None:
                    p["dx"] = 0.0
                    p["dy"] = 0.0
                    p["angle"] = 0.0
                else:
                    p["dx"] = p["x"] - prev_x
                    p["dy"] = p["y"] - prev_y
                    p["angle"] = np.arctan2(p["dy"], p["dx"])
                prev_x, prev_y = p["x"], p["y"]
                # pressure إذا موجود في البيانات، أو 1.0 افتراضي
                p["pressure"] = p.get("pressure", 1.0)
        return {"gesture_id": gesture.get("id"), "character": gesture.get("character"),
                "duration_ms": gesture.get("duration_ms",0), "frames": frames,
                "frame_count": len(frames)}

    def load_all_gestures(self) -> List[Dict]:
        page = 1
        all_gestures = []
        while True:
            url = f"{self.api_url}?page={page}&per_page={self.per_page}"
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            gestures = data.get("data", [])
            if not gestures: break
            for g in gestures:
                processed = self._process_gesture(g)
                if processed:
                    all_gestures.append(processed)
            page += 1
        return all_gestures
