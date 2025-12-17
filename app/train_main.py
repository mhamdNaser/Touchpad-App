# train_main.py
import sys
import traceback
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
import cv2
import kagglehub

from app.services.gesture_data_loader import GestureDataLoader
from app.services.gesture_preprocessor import GesturePreprocessor
from app.services.feature_extractor import GestureFeatureExtractor
from app.services.training_pipeline import GestureTrainer
from app.services.gesture_utils import summarize_processed, summarize_csv


def main(mode="train"):
    try:
        print("=" * 60)
        print(f"🎯 Arabic Gesture Recognition - Mode: {mode.upper()}")
        print("=" * 60)

        if mode == "save_features":
            csv_path = "gestures_features.csv"
            print(f"\n💾 Loading gestures from API and saving features to CSV: {csv_path} ...")

            data_loader = GestureDataLoader(api_url="https://api.sydev.site/api/gestures")
            data_loader.load_api_data()
            gestures_data = data_loader.parse_data()

            preprocessor = GesturePreprocessor(resample_frames=50)
            processed = preprocessor.preprocess(gestures_data)
            processed_gestures_for_extractor = []
            for i, g in enumerate(gestures_data):
                pts = processed[i]
                pts_list = [{"x": float(x), "y": float(y)} for x, y in pts]
                processed_gestures_for_extractor.append(
                    {"id": g["id"], "character": g["character"], "points": pts_list}
                )

            extractor = GestureFeatureExtractor(
                image_size=64, thickness=1.5, channels=("stroke", "velocity"), resample_frames=50
            )
            extractor.save_features_to_csv(processed_gestures_for_extractor, csv_path)
            print(f"✅ Saved feature images to {csv_path}")
            return

        elif mode == "save_csv":
            csv_path = "gestures_flat.csv"
            print(f"\n💾 Loading gestures from API and saving to CSV: {csv_path} ...")

            data_loader = GestureDataLoader(api_url="https://api.sydev.site/api/gestures")
            data_loader.load_api_data()
            data_loader.parse_data()
            data_loader.save_to_csv_flat(csv_path)

            print(f"✅ Saved all gestures to {csv_path}")
            return

        elif mode == "summary":
            csv_path = "gestures_flat.csv"
            summarize_csv(csv_path)
            return

        elif mode == "train":

            csv_path = "gestures_flat.csv"
            print(f"\n📄 Loading gestures from CSV: {csv_path}")
            if not os.path.exists(csv_path):
                raise FileNotFoundError("CSV not found!")

            df = pd.read_csv(csv_path)

            # استخراج الإيماءات
            gestures_data = []
            max_idx = max([int(col[1:]) for col in df.columns if col.startswith("x")], default=0)

            for _, row in df.iterrows():
                gesture_id = row["gesture_id"]
                character = row["character"]
                points = []

                for idx in range(1, max_idx + 1):
                    x_key = f"x{idx}"
                    y_key = f"y{idx}"
                    p_key = f"pressure{idx}"

                    if x_key not in df.columns:
                        break

                    x = row[x_key]
                    y = row[y_key]
                    p = row[p_key] if p_key in df.columns else 0.0

                    if pd.isna(x) or pd.isna(y):
                        break

                    points.append({"x": float(x), "y": float(y), "pressure": float(p)})

                gestures_data.append(
                    {"id": int(gesture_id), "character": character, "points": points}
                )

            print(f"✅ Loaded gestures: {len(gestures_data)}")

            # تحويل الإيماءات لصور
            extractor = GestureFeatureExtractor(image_size=64, thickness=1.5,
                                                channels=("stroke", "velocity"), resample_frames=50)

            X_gestures = []
            y_gestures = []

            from tqdm import tqdm
            for g in tqdm(gestures_data, desc="Converting gestures to images"):
                img = extractor.gesture_to_image(g["points"])
                X_gestures.append(img)
                y_gestures.append(g["character"])

            X_gestures = np.array(X_gestures, dtype=np.float32)
            y_gestures = np.array(y_gestures, dtype=str)

            # ضمان عدد القنوات = 1
            if X_gestures.ndim == 4 and X_gestures.shape[-1] > 1:
                X_gestures = X_gestures[..., :1]

            print(f"🖼️ Gesture images shape: {X_gestures.shape}")

            # تدريب الموديل
            trainer = GestureTrainer(image_size=(64, 64))
            model, history = trainer.train(X_gestures, y_gestures)

    except Exception as e:
        print(f"❌ Error in {mode} mode: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    mode_arg = sys.argv[1].lower() if len(sys.argv) > 1 else "train"
    main(mode_arg)
