import requests
from typing import List, Dict
from app.services.preprocess import Preprocessor

class DataLoader:
    def __init__(self, api_url: str = "https://api.sydev.site/api/gestures"):
        self.api_url = api_url
        self.preprocessor = Preprocessor()

    def load_gestures_data(self, characters: List[str], limit_per_char: int = 200) -> List[Dict]:
        """
        تحميل بيانات الإيماءات من API مع معالجة المفتاح "data"
        """
        gestures_data = []
        print(f"Fetching gestures from API: {self.api_url}")

        try:
            response = requests.get(self.api_url)
            response.raise_for_status()

            all_gestures = response.json()

            # ✅ استخدم مفتاح "data" إذا موجود
            if isinstance(all_gestures, dict) and "data" in all_gestures:
                all_gestures = all_gestures["data"]

            print(f"✅ API returned {len(all_gestures)} gestures")

            # تصفية وتحويل لكل حرف
            for char in characters:
                char_gestures = [g for g in all_gestures if g.get('character') == char]
                char_gestures = char_gestures[:limit_per_char]

                for gesture in char_gestures:
                    gesture_data = self.preprocessor.process_gesture(gesture)
                    if gesture_data:
                        gestures_data.append(gesture_data)

                print(f"✅ Loaded {len(char_gestures)} gestures for '{char}'")

        except Exception as e:
            print(f"❌ Error fetching data from API: {e}")

        return gestures_data





# # import pandas as pd
# from sqlalchemy.orm import Session, selectinload
# from app.models.gesture import Gesture
# from app.models.frame import Frame
# from app.models.point import Point
# from typing import List, Dict
# from app.services.preprocess import Preprocessor

# ###############################################
# ##  Download gesture data from the database  ##
# ###############################################
# class DataLoader:
#     def __init__(self, db: Session):
#         self.db = db
#         self.preprocessor = Preprocessor()

#     def load_gestures_data(self, characters: List[str], limit_per_char: int = 200, batch_size: int = 500) -> List[Dict]:
#         """
#         تحميل بيانات الإيماءات من قاعدة البيانات على دفعات (Batch-wise loading)
#         """
#         gestures_data = []

#         for char in characters:
#             offset = 0
#             loaded_count = 0

#             while loaded_count < limit_per_char:
#                 gestures = (
#                     self.db.query(Gesture)
#                     .options(selectinload(Gesture.frames).selectinload(Frame.points))
#                     .filter(Gesture.character == char)
#                     .offset(offset)
#                     .limit(batch_size)
#                     .all()
#                 )

#                 if not gestures:
#                     break 

#                 for gesture in gestures:
#                     if loaded_count >= limit_per_char:
#                         break
#                     gesture_data = self.preprocessor.process_gesture(gesture)
#                     if gesture_data:
#                         gestures_data.append(gesture_data)
#                         loaded_count += 1
#                         print(f"Loaded {loaded_count}/{limit_per_char} gestures for '{char}'")

#                 offset += batch_size

#         return gestures_data