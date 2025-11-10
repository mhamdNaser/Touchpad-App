from typing import Dict, Any, List

#################################
##   Data Preprocessing Step   ##
#################################
class Preprocessor:
    def __init__(self):
        pass

    def process_gesture(self, gesture: Dict[str, Any]) -> Dict:
        """
        معالجة إيماءة واحدة قادمة من API وتحويلها إلى تنسيق مناسب للتدريب
        """
        try:
            # تأكيد وجود الحقول الأساسية
            gesture_id = gesture.get("id")
            character = gesture.get("character")
            duration_ms = gesture.get("duration_ms", 0)
            frame_count = gesture.get("frame_count", 0)
            frames = gesture.get("frames", [])

            if not frames:
                print(f"⚠️ Gesture {gesture_id} has no frames, skipped.")
                return None

            # فرز الفريمات حسب frame_id أو id حسب المتاح
            sorted_frames = sorted(frames, key=lambda f: f.get("frame_id", f.get("id", 0)))

            frames_data: List[Dict] = []

            for frame in sorted_frames:
                frame_id = frame.get("frame_id", frame.get("id"))
                timestamp = frame.get("timestamp", frame.get("ts"))
                points = frame.get("points", [])

                frame_data = {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "points": []
                }

                for pt in points:
                    # بعض الـ API قد تحتوي أسماء مختلفة أو ناقصة
                    frame_data["points"].append({
                        "x": pt.get("x", 0.0),
                        "y": pt.get("y", 0.0),
                        "state": pt.get("state", "unknown"),
                        "pressure": pt.get("pressure", 0.0)
                    })

                frames_data.append(frame_data)

            return {
                "gesture_id": gesture_id,
                "character": character,
                "frames": frames_data,
                "duration_ms": duration_ms,
                "frame_count": frame_count
            }

        except Exception as e:
            print(f"❌ Error processing gesture: {e}")
            return None



# from typing import Dict
# from app.models.gesture import Gesture

# #################################
# ##   Data Preprocessing Step   ##
# #################################
# class Preprocessor:
#     def __init__(self):
#         pass

#     def process_gesture(self, gesture: Gesture) -> Dict:
#         """
#         معالجة إيماءة واحدة وتحويلها إلى تنسيق مناسب للتدريب
#         """
#         frames_data = []
#         sorted_frames = sorted(gesture.frames, key=lambda x: x.frame_id)

#         for frame in sorted_frames:
#             frame_data = {
#                 "frame_id": frame.frame_id,
#                 "timestamp": frame.timestamp,
#                 "points": []
#             }

#             for point in frame.points:
#                 frame_data["points"].append({
#                     "x": point.x,
#                     "y": point.y,
#                     "state": point.state,
#                     "pressure": point.pressure or 0.0
#                 })

#             frames_data.append(frame_data)

#         return {
#             "gesture_id": gesture.id,
#             "character": gesture.character,
#             "frames": frames_data,
#             "duration_ms": gesture.duration_ms,
#             "frame_count": gesture.frame_count
#         }
