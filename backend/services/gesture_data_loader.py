import requests
from typing import List, Dict, Any
import numpy as np


# ================================
# Helpers
# ================================

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


# =====================================
# Main Loader Class
# =====================================

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



# import requests
# from typing import List, Dict, Any

# class GestureDataLoader:
#     def __init__(self, api_url: str = "https://api.sydev.site/api/gestures", per_page: int = 50):
#         self.api_url = api_url
#         self.per_page = per_page
#         self.session = requests.Session()
#         self.session.timeout = 30
#         self.allowed_states = ("down", "move")

#     def _frame_sort_key(self, frame: Dict[str, Any]):
#         return frame.get("timestamp") or frame.get("ts") or frame.get("frame_id") or 0

#     def _process_gesture(self, gesture: Dict[str, Any]) -> Dict:
#         """تحويل Gesture إلى شكل مرتب جاهز للـ Feature Extraction في كلاس آخر"""
#         try:
#             frames_raw = gesture.get("frames", [])
#             points_raw = gesture.get("points", [])

#             frames: List[Dict] = []

#             if frames_raw:
#                 sorted_frames = sorted(frames_raw, key=self._frame_sort_key)
#                 prev_ts = None
#                 for f in sorted_frames:
#                     ts = f.get("timestamp") or f.get("ts") or 0
#                     delta = f.get("delta_ms", 0)
#                     if not delta and prev_ts is not None:
#                         delta = ts - prev_ts
#                     prev_ts = ts

#                     pts = f.get("points", []) or f.get("raw_payload", {}).get("points", [])
#                     clean_pts = [
#                         {
#                             "x": p.get("x", 0.0),
#                             "y": p.get("y", 0.0),
#                             "pressure": p.get("pressure", 0.0),
#                             "angle": p.get("angle", 0.0),
#                             "vx": p.get("vx", 0.0),
#                             "vy": p.get("vy", 0.0),
#                             "dx": p.get("dx", 0.0),
#                             "dy": p.get("dy", 0.0)
#                         }
#                         for p in pts if p.get("state") in self.allowed_states
#                     ]
#                     frames.append({
#                         "frame_id": f.get("frame_id", f.get("id")),
#                         "timestamp": ts,
#                         "delta_ms": delta,
#                         "points": clean_pts
#                     })

#             elif points_raw:
#                 temp = {}
#                 for p in points_raw:
#                     fid = p.get("frame_id")
#                     if fid not in temp:
#                         temp[fid] = {
#                             "frame_id": fid,
#                             "timestamp": p.get("timestamp", 0),
#                             "delta_ms": p.get("delta_ms", 0),
#                             "points": []
#                         }
#                     if p.get("state") in self.allowed_states:
#                         temp[fid]["points"].append({
#                             "x": p.get("x", 0.0),
#                             "y": p.get("y", 0.0),
#                             "pressure": p.get("pressure", 0.0),
#                             "angle": p.get("angle", 0.0),
#                             "vx": p.get("vx", 0.0),
#                             "vy": p.get("vy", 0.0),
#                             "dx": p.get("dx", 0.0),
#                             "dy": p.get("dy", 0.0)
#                         })
#                 frames = sorted(list(temp.values()), key=self._frame_sort_key)
#             else:
#                 return None

#             return {
#                 "gesture_id": gesture.get("id"),
#                 "character": gesture.get("character"),
#                 "duration_ms": gesture.get("duration_ms", 0),
#                 "frames": frames,
#                 "frame_count": len(frames)
#             }
#         except Exception:
#             return None

#     def load_all_gestures(self) -> List[Dict]:
#         """
#         تحميل جميع الإيماءات باستخدام Pagination داخليًا
#         - يحافظ على اسم الدالة الأصلي
#         - جاهز للاستخدام مع كلاس الـ Feature Extraction
#         """
#         page = 1
#         all_gestures = []

#         while True:
#             url = f"{self.api_url}?page={page}&per_page={self.per_page}"
#             response = self.session.get(url)
#             response.raise_for_status()
#             data = response.json()
#             gestures = data.get("data", [])

#             if not gestures:
#                 break

#             for g in gestures:
#                 processed = self._process_gesture(g)
#                 if processed:
#                     all_gestures.append(processed)
#             page += 1

#         return all_gestures


# # -------------------------
# # مثال تشغيل
# # -------------------------
# if __name__ == "__main__":
#     loader = GestureDataLoader(per_page=50)
#     gestures = loader.load_all_gestures()
#     print(f"Loaded {len(gestures)} gestures")