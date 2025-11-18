# app/train_main.py
import sys
import os
import traceback

# ✅ استيراد الملفات الصحيحة
from app.services.gesture_data_loader import GestureDataLoader
from app.services.advanced_feature_extractor import AdvancedFeatureExtractor
from app.services.training_pipeline import TrainingPipeline


def main(mode="train"):
    try:
        print("=" * 60)
        print(f"🎯 Starting Arabic Gesture Recognition - Mode: {mode.upper()}")
        print("=" * 60)

        # تحميل البيانات مرة واحدة لجميع الأوضاع
        data_loader = GestureDataLoader(api_url="https://api.sydev.site/api/gestures")
        gestures_data = data_loader.load_all_gestures()
        print(f"✅ Loaded {len(gestures_data)} gestures from API")

        # =====  التدريب =====
        if mode == "train":
            print("\n🏋️ Starting Model Training...")
            pipeline = TrainingPipeline(max_timesteps=50)
            result = pipeline.train_model()
            print(f"✅ Training completed. Test accuracy: {result['test_accuracy']:.3f}")
        
        elif mode == "analyze":
            print("\n📊 Extracting Advanced Feature CSV...")
            extractor = AdvancedFeatureExtractor(max_timesteps=200, verbose=True)
            extractor.save_gestures_to_csv(gestures_data, out_csv="ADVANCED_features.csv")
            print("✅ Advanced feature extraction completed.")

            print("\n📈 Plotting Feature Variance...")
            extractor.plot_feature_variance(gestures_data)

        else:
            print(f"❌ Unknown mode '{mode}'.")
            print("💡 Use: train, test, or help")

    except Exception as e:
        print(f"❌ Error in {mode} mode: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # معالجة وسيطات سطر الأوامر
    if len(sys.argv) > 1:
        mode_arg = sys.argv[1].lower()
    else:
        mode_arg = "train"
    main(mode_arg)
