# app/train_main.py
import sys
import os
import traceback

# ✅ استيراد الملفات الصحيحة
from app.services.gesture_data_loader import GestureDataLoader
from app.other_services.feature_generator import StatisticalFeatureGenerator
from app.services.training_pipeline import TrainingPipeline
from app.services.test_model import main as test_main


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

        # =====  الاختبار =====
        elif mode == "test":
            print("\n🧪 Starting Model Testing...")
            
            # التحقق من وجود الملفات المطلوبة
            required_files = [
                "arabic_gesture_cnn_best.keras", 
                "label_encoder.pkl",
                "X_test.pkl",
                "y_test.pkl"
            ]
            
            missing_files = [f for f in required_files if not os.path.exists(f)]
            if missing_files:
                print(f"❌ Missing required files: {missing_files}")
                print("💡 Please run training first: python -m app.train_main train")
                return
            
            test_main()
        
        elif mode == "extract_adv":
            print("\n📊 Extracting Advanced Feature CSV...")
            generator = StatisticalFeatureGenerator(max_timesteps=200, verbose=True)
            generator.process_gestures(
                gestures_data,
                out_csv="ADVANCED_features.csv"
            )
            print("✅ Advanced feature extraction completed.")

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
