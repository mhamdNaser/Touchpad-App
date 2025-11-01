from typing import Dict
from app.models.gesture import Gesture

#################################
##   Data Preprocessing Step   ##
#################################
class Preprocessor:
    def __init__(self):
        pass

    def process_gesture(self, gesture: Gesture) -> Dict:
        """
        معالجة إيماءة واحدة وتحويلها إلى تنسيق مناسب للتدريب
        """
        frames_data = []
        sorted_frames = sorted(gesture.frames, key=lambda x: x.frame_id)

        for frame in sorted_frames:
            frame_data = {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "points": []
            }

            for point in frame.points:
                frame_data["points"].append({
                    "x": point.x,
                    "y": point.y,
                    "state": point.state,
                    "pressure": point.pressure or 0.0
                })

            frames_data.append(frame_data)

        return {
            "gesture_id": gesture.id,
            "character": gesture.character,
            "frames": frames_data,
            "duration_ms": gesture.duration_ms,
            "frame_count": gesture.frame_count
        }
