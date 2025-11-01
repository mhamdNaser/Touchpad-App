# import pandas as pd
from sqlalchemy.orm import Session, selectinload
from app.models.gesture import Gesture
from app.models.frame import Frame
from app.models.point import Point
from typing import List, Dict
from app.services.preprocess import Preprocessor

###############################################
##  Download gesture data from the database  ##
###############################################
class DataLoader:
    def __init__(self, db: Session):
        self.db = db
        self.preprocessor = Preprocessor()

    def load_gestures_data(self, characters: List[str], limit_per_char: int = 200, batch_size: int = 500) -> List[Dict]:
        """
        تحميل بيانات الإيماءات من قاعدة البيانات على دفعات (Batch-wise loading)
        """
        gestures_data = []

        for char in characters:
            offset = 0
            loaded_count = 0

            while loaded_count < limit_per_char:
                gestures = (
                    self.db.query(Gesture)
                    .options(selectinload(Gesture.frames).selectinload(Frame.points))
                    .filter(Gesture.character == char)
                    .offset(offset)
                    .limit(batch_size)
                    .all()
                )

                if not gestures:
                    break 

                for gesture in gestures:
                    if loaded_count >= limit_per_char:
                        break
                    gesture_data = self.preprocessor.process_gesture(gesture)
                    if gesture_data:
                        gestures_data.append(gesture_data)
                        loaded_count += 1

                offset += batch_size

        return gestures_data