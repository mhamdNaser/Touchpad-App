# app/services/gesture_data_loader.py
import requests
from typing import List, Dict, Any

class GestureDataLoader:
    def __init__(self, api_url: str = "https://api.sydev.site/api/gestures"):
        self.api_url = api_url

    # هي الدالة الي لح بتقوم بمعالجة البيانات وتنظيمها بس طبعا لازم تستلمها ك دكشنري 
    def _process_gesture(self, gesture: Dict[str, Any]) -> Dict:
        try:
            gesture_id = gesture.get("id")
            character = gesture.get("character")
            duration_ms = gesture.get("duration_ms", 0)
            frame_count = gesture.get("frame_count", 0)

            # نتحقق من وجود فريمز او بوينس 
            points = gesture.get("points", [])
            frames = gesture.get("frames", [])

            frames_data: List[Dict] = []

            # ✅ الحالة 1 : الايماءات فيها فريمات
            if frames:
                sorted_frames = sorted(frames, key=lambda f: f.get("timestamp", f.get("frame_id", f.get("id", 0))))
                for frame in sorted_frames:
                    frame_id = frame.get("frame_id", frame.get("id"))
                    timestamp = frame.get("timestamp", frame.get("ts"))
                    delta_ms = frame.get("delta_ms", 0)

                    # جلب النقاط من points أو raw_payload.points
                    pts = frame.get("points", [])
                    if not pts and "raw_payload" in frame:
                        pts = frame["raw_payload"].get("points", [])

                    frame_data = {
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "delta_ms": delta_ms,
                        "points": []
                    }

                    for pt in pts:
                        frame_data["points"].append({
                            "x": pt.get("x", 0.0),
                            "y": pt.get("y", 0.0),
                            "state": pt.get("state", "unknown"),
                            "pressure": pt.get("pressure", 0.0),
                            "angle": pt.get("angle", 0.0),
                            "vx": pt.get("vx", 0.0),
                            "vy": pt.get("vy", 0.0),
                            "dx": pt.get("dx", 0.0),
                            "dy": pt.get("dy", 0.0)
                        })

                    frames_data.append(frame_data)
                    
            # الحالة 2 : الايماءا فيها بوينت
            elif points:
                frame_data = {
                    "frame_id": gesture_id,
                    "timestamp": points[0].get("timestamp", 0) if points else 0,
                    "delta_ms": points[0].get("delta_ms", 0) if points else 0,
                    "points": []
                }

                for pt in points:
                    frame_data["points"].append({
                        "x": pt.get("x", 0.0),
                        "y": pt.get("y", 0.0),
                        "state": pt.get("state", "unknown"),
                        "pressure": pt.get("pressure", 0.0),
                        "angle": pt.get("angle", 0.0),
                        "vx": pt.get("vx", 0.0),
                        "vy": pt.get("vy", 0.0),
                        "dx": pt.get("dx", 0.0),
                        "dy": pt.get("dy", 0.0)
                    })

                frames_data.append(frame_data)

            else:
                print(f"⚠️ Gesture {gesture_id} has no frames or points")
                return None

            return {
                "gesture_id": gesture_id,
                "character": character,
                "frames": frames_data,
                "duration_ms": duration_ms,
                "frame_count": frame_count or len(frames_data)
            }

        except Exception as e:
            print(f"❌ Error processing gesture {gesture.get('id')}: {e}")
            return None

    def load_all_gestures(self) -> List[Dict]:
        
        gestures_data: List[Dict] = []

        try:
            response = requests.get(self.api_url)
            response.raise_for_status()

            all_gestures = response.json()

            # استخدام مفتاح للداتا لو كان موجود بس بحالتنا قمنا بإعداد داتا خاصة فينا 
            if isinstance(all_gestures, dict) and "data" in all_gestures:
                all_gestures = all_gestures["data"]

            # معالجة كل إيماءة مباشرة بعد الجلب
            for gesture in all_gestures:
                processed = self._process_gesture(gesture)
                if processed:
                    gestures_data.append(processed)

            print(f"✅ Total processed gestures: {len(gestures_data)}")


        except Exception as e:
            print(f"❌ Error fetching data from API: {e}")

        return gestures_data