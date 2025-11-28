import sys
import traceback
import numpy as np
import pandas as pd  # لإظهار ملخص CSV

from app.services.gesture_data_loader import GestureDataLoader
from app.services.gesture_preprocessor import GesturePreprocessor
from app.services.feature_extractor import GestureFeatureExtractor
from app.services.training_pipeline import GestureTrainer
from app.services.gesture_utils import summarize_processed, summarize_csv


def main(mode="train"):
    try:
        print("="*60)
        print(f"🎯 Arabic Gesture Recognition - Mode: {mode.upper()}")
        print("="*60)

        if mode == "save_csv":
            csv_path = "gestures_flat.csv"
            print(f"\n💾 Loading gestures from API and saving to CSV: {csv_path} ...")

            data_loader = GestureDataLoader(api_url="https://api.sydev.site/api/gestures")
            data_loader.load_api_data()
            data_loader.parse_data()
            data_loader.save_to_csv_flat(csv_path)

            print(f"✅ Saved all gestures to {csv_path}")
            return

        elif mode == "summary":
            csv_path = "gestures_flat.csv"  # نفس الملف المخزن
            summarize_csv(csv_path)
            return

        elif mode == "summary_processed":
            data_loader = GestureDataLoader(api_url="https://api.sydev.site/api/gestures")
            data_loader.load_api_data()
            gestures_data = data_loader.parse_data()
            preprocessor = GesturePreprocessor()
            processed_gestures = preprocessor.preprocess(gestures_data)
            summarize_processed(gestures_data, processed_gestures)
            return
        
        # ===== باقي الأوضاع train و preview كما قبل =====
        print("\n📥 Loading gestures data from JSON...")
        data_loader = GestureDataLoader(json_path="data.json")
        gestures_data = data_loader.parse_data()
        print(f"✅ Loaded {len(gestures_data)} gestures")

        preprocessor = GesturePreprocessor()
        processed_gestures = preprocessor.preprocess(gestures_data)
        print(f"✅ Preprocessed gestures shape: {processed_gestures.shape}")

        extractor = GestureFeatureExtractor()
        features = np.array([extractor.extract_features(g['points']) for g in gestures_data])
        print(f"✅ Features extracted: {features.shape}")

        if mode == "train":
            print("\n🏋️ Training the model...")
            input_shape = (features.shape[1], 1)
            num_classes = len(set([g['character'] for g in gestures_data]))
            trainer = GestureTrainer(input_shape=input_shape, num_classes=num_classes)
            trainer.train(features.reshape(features.shape[0], features.shape[1], 1),
                          [g['character'] for g in gestures_data])
            print(f"✅ Training completed!")

        elif mode == "preview":
            print("\n🔍 Previewing gestures data...")
            for i, g in enumerate(gestures_data[:5]):
                print(f"Gesture {i+1}: character='{g['character']}', frames={len(g['points'])}")

        else:
            print(f"❌ Unknown mode '{mode}'.")
            print("💡 Use: train, preview, save_csv, summary")

    except Exception as e:
        print(f"❌ Error in {mode} mode: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode_arg = sys.argv[1].lower()
    else:
        mode_arg = "train"
    main(mode_arg)
