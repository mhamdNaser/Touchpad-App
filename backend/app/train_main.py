# app/train_main.py
import sys
import os
import traceback
from app.services.gesture_data_loader import GestureDataLoader
from app.services.features_visualizer import FeatureEngineerVisualizer
from app.services.training_pipeline import TrainingPipeline
from app.services.test_model import main as test_main

def main(mode="analyze"):
    """
    🚀 البرنامج الرئيسي للتدريب والاختبار
    أوامر التشغيل:
    python -m app.train_main analyze
    python -m app.train_main train  
    python -m app.train_main test
    python -m app.train_main retrain  # ⭐ جديد
    """
    
    try:
        print("=" * 60)
        print(f"🎯 Starting Arabic Gesture Recognition - Mode: {mode.upper()}")
        print("=" * 60)

        # تحميل البيانات مرة واحدة لجميع الأوضاع
        data_loader = GestureDataLoader(api_url="https://api.sydev.site/api/gestures")
        gestures_data = data_loader.load_all_gestures()
        
        if not gestures_data:
            print("❌ No data loaded. Exiting.")
            return

        print(f"✅ Loaded {len(gestures_data)} gestures from API")

        # =====  التحليل =====
        if mode == "analyze":
            print("\n📊 Starting Data Analysis...")
            feature_engineer = FeatureEngineerVisualizer(max_timesteps=150)
            
            # تحليل توزيع الميزات
            feature_engineer.plot_feature_distribution(gestures_data)
            
            # تحليل إضافي للبيانات
            print("\n🔍 Additional Data Analysis...")
            characters = [gesture['character'] for gesture in gestures_data]
            unique_chars, counts = np.unique(characters, return_counts=True)
            print(f"📈 Character distribution: {dict(zip(unique_chars, counts))}")
            
            # تحليل عدد الإطارات
            frame_counts = []
            for gesture in gestures_data:
                frames = gesture.get('frames', [])
                if not frames and 'points' in gesture:
                    frames = [gesture]  # صيغة قديمة
                frame_counts.append(len(frames))
            
            print(f"📊 Frame statistics - Min: {min(frame_counts)}, Max: {max(frame_counts)}, Avg: {sum(frame_counts)/len(frame_counts):.1f}")

        # =====  التدريب =====
        elif mode == "train":
            print("\n🏋️ Starting Model Training...")
            pipeline = TrainingPipeline(max_timesteps=150)
            result = pipeline.train_model()
            print(f"✅ Training completed. Test accuracy: {result['test_accuracy']:.3f}")

        # =====  إعادة التدريب (جديد) =====
        elif mode == "retrain":
            print("\n🔄 Starting Model Retraining with Fixed Preprocessing...")
            pipeline = TrainingPipeline(max_timesteps=150)
            
            # استخدام دالة إعادة التدريب الجديدة
            if hasattr(pipeline, 'retrain_with_fixed_scaling'):
                result = pipeline.retrain_with_fixed_scaling(gestures_data)
            else:
                # إذا لم تكن الدالة موجودة، استخدم التدريب العادي
                print("⚠️  Using standard training (retrain method not available)")
                result = pipeline.train_model()
                
            print(f"✅ Retraining completed. Test accuracy: {result['test_accuracy']:.3f}")

        # =====  الاختبار =====
        elif mode == "test":
            print("\n🧪 Starting Model Testing...")
            
            # التحقق من وجود الملفات المطلوبة
            required_files = [
                "arabic_gesture_cnn_best.h5", 
                "scaler.pkl", 
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

        # =====  وضع المساعدة =====
        elif mode == "help":
            print("""
                    📖 Available Commands:
                    python -m app.train_main analyze   - تحليل البيانات وتوزيع الميزات
                    python -m app.train_main train     - تدريب النموذج من الصفر  
                    python -m app.train_main retrain   - إعادة التدريب مع إصلاح المعايرة
                    python -m app.train_main test      - اختبار النموذج المدرب
                    python -m app.train_main help      - عرض هذه المساعدة
            """)

        else:
            print(f"❌ Unknown mode '{mode}'.")
            print("💡 Use: analyze, train, retrain, test, or help")

    except Exception as e:
        print(f"❌ Error in {mode} mode: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # معالجة وسيطات سطر الأوامر
    if len(sys.argv) > 1:
        mode_arg = sys.argv[1].lower()
    else:
        mode_arg = "help"  # عرض المساعدة افتراضياً
    
    # تحميل numpy فقط إذا needed
    if mode_arg == "analyze":
        import numpy as np
        
    main(mode_arg)