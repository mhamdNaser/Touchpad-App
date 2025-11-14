import requests
from typing import List, Dict, Any
import time

class GestureDataLoader:
    def __init__(self, api_url: str = "https://api.sydev.site/api/gestures"):
        self.api_url = api_url
        self.session = requests.Session()
        self.session.timeout = 30
        self.allowed_states = ("down", "move")

    # -------------------------------------------------------------
    # Internal: sort key
    # -------------------------------------------------------------
    def _frame_sort_key(self, frame: Dict[str, Any]):
        return (
            frame.get("timestamp") or
            frame.get("ts") or
            frame.get("frame_id") or
            frame.get("id") or
            0
        )

    # -------------------------------------------------------------
    # Internal: process a single gesture
    # -------------------------------------------------------------
    def _process_gesture(self, gesture: Dict[str, Any]) -> Dict:
        try:
            gesture_id = gesture.get("id")

            frames_raw = gesture.get("frames", [])
            points_raw = gesture.get("points", [])

            frames: List[Dict] = []

            # Case A: gesture has dedicated frames list
            if frames_raw:
                sorted_frames = sorted(frames_raw, key=self._frame_sort_key)
                prev_ts = None

                for f in sorted_frames:
                    ts = f.get("timestamp") or f.get("ts") or 0
                    delta = f.get("delta_ms", 0)

                    # compute delta if missing
                    if (not delta) and prev_ts is not None:
                        delta = ts - prev_ts
                    prev_ts = ts

                    pts = f.get("points", [])
                    if not pts and "raw_payload" in f:
                        pts = f["raw_payload"].get("points", [])

                    clean_pts = []
                    for p in pts:
                        if p.get("state") in self.allowed_states:
                            clean_pts.append({
                                "x": p.get("x", 0.0),
                                "y": p.get("y", 0.0),
                                "pressure": p.get("pressure", 0.0),
                                "angle": p.get("angle", 0.0),
                                "vx": p.get("vx", 0.0),
                                "vy": p.get("vy", 0.0),
                                "dx": p.get("dx", 0.0),
                                "dy": p.get("dy", 0.0)
                            })

                    frames.append({
                        "frame_id": f.get("frame_id", f.get("id")),
                        "timestamp": ts,
                        "delta_ms": delta,
                        "points": clean_pts
                    })

            # Case B: raw points only, grouped by frame_id
            elif points_raw:
                temp = {}
                for p in points_raw:
                    fid = p.get("frame_id")
                    if fid not in temp:
                        temp[fid] = {
                            "frame_id": fid,
                            "timestamp": p.get("timestamp", 0),
                            "delta_ms": p.get("delta_ms", 0),
                            "points": []
                        }

                    if p.get("state") in self.allowed_states:
                        temp[fid]["points"].append({
                            "x": p.get("x", 0.0),
                            "y": p.get("y", 0.0),
                            "pressure": p.get("pressure", 0.0),
                            "angle": p.get("angle", 0.0),
                            "vx": p.get("vx", 0.0),
                            "vy": p.get("vy", 0.0),
                            "dx": p.get("dx", 0.0),
                            "dy": p.get("dy", 0.0)
                        })

                frames = list(temp.values())
                frames.sort(key=self._frame_sort_key)

            else:
                return None

            return {
                "gesture_id": gesture.get("id"),
                "character": gesture.get("character"),
                "duration_ms": gesture.get("duration_ms", 0),
                "frames": frames,
                "frame_count": len(frames)
            }

        except Exception:
            return None

    # -------------------------------------------------------------
    # Public: load all gestures
    # -------------------------------------------------------------
    def load_all_gestures(self) -> List[Dict]:
        try:
            response = self.session.get(self.api_url)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict):
                if "data" in data:
                    data = data["data"]
                elif "gestures" in data:
                    data = data["gestures"]

            if not isinstance(data, list):
                return []

            processed = []
            for g in data:
                p = self._process_gesture(g)
                if p:
                    processed.append(p)

            return processed

        except Exception:
            return []


# optional runnable entry
if __name__ == "__main__":
    ld = GestureDataLoader()
    gestures = ld.load_all_gestures()
    print(f"Loaded: {len(gestures)} gestures")