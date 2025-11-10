import requests
from typing import List, Dict
from app.services.preprocess import Preprocessor

class DataLoader:
    def __init__(self, api_url: str = "https://api.sydev.site/api/gestures"):
        """
        تحميل بيانات الإيماءات من API خارجي بدل قاعدة البيانات.
        """
        self.api_url = api_url
        self.preprocessor = Preprocessor()

    def load_gestures_data(self, characters: List[str], limit_per_char: int = 200) -> List[Dict]:
        """
        تحميل بيانات الإيماءات من API وتصفيتها حسب الحروف المطلوبة.
        """
        gestures_data = []

        try:
            # 1️⃣ جلب جميع الجستشر من الـ API
            print(f"Fetching gestures from API: {self.api_url}")
            response = requests.get(self.api_url)
            response.raise_for_status()

            data = response.json()

            # بعض الـ API ترجع {"data": [...]}، لذا نتحقق
            gestures = data.get("data", data)

            # 2️⃣ تصفية الجستشر حسب الحروف المطلوبة
            for char in characters:
                filtered = [g for g in gestures if g.get("character") == char]
                limited = filtered[:limit_per_char]

                for i, gesture in enumerate(limited, start=1):
                    gesture_data = self.preprocessor.process_gesture(gesture)
                    if gesture_data:
                        gestures_data.append(gesture_data)
                        print(f"Loaded {i}/{limit_per_char} gestures for '{char}'")

        except requests.exceptions.RequestException as e:
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