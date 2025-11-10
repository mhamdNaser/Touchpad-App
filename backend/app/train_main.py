# app/train_main.py
import sys
import pickle
import json
import numpy as np
import pandas as pd
from app.services.data_loader import DataLoader
from app.services.features import FeatureEngineer
from app.services.training_pipeline import TrainingPipeline
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report

def main(mode="analyze"):
    """
    mode: 
        "analyze" -> تحليل الميزات فقط
        "train"   -> تدريب النموذج
        "test"    -> اختبار النموذج
    """
    characters = ["ا", "ب", "ت"]

    # 1️⃣ تحميل البيانات من API
    loader = DataLoader(api_url="https://api.sydev.site/api/gestures")
    features_extractor = FeatureEngineer(max_timesteps=200)
    data = loader.load_gestures_data(characters, limit_per_char=50)
    print(f"\n✅ Loaded {len(data)} gestures\n")

    if len(data) == 0:
        print("❌ No gestures loaded. Check API or character list.")
        return

    # 2️⃣ تحليل الميزات وعرض الجدول
    aggregated_features = features_extractor.aggregate_by_character(data)
    df = features_extractor.show_feature_table(aggregated_features)
    df.to_csv("gesture_features_analysis.csv", encoding="utf-8-sig")
    print("💾 Saved CSV: gesture_features_analysis.csv\n")

    # 3️⃣ تشغيل التدرب أو الاختبار بناءً على الـ mode
    if mode == "train":
        print("🚀 Starting training pipeline...")
        pipeline = TrainingPipeline()
        result = pipeline.train_model(characters)
        print(f"\n✅ Training completed. Test accuracy: {result['test_accuracy']:.3f}")

    elif mode == "test":
        print("🧪 Starting test pipeline...")

        # استخراج الميزات
        X, y = features_extractor.extract_features(data)
        print(f"🔹 Feature shape: {X.shape}, Number of classes: {len(np.unique(y))}")

        # تحميل النموذج المحفوظ
        model = load_model("arabic_gesture_cnn_final.h5")
        with open("scaler.pkl", "rb") as f:
            features_extractor.scaler = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            features_extractor.label_encoder = pickle.load(f)

        # تحويل التسميات إلى one-hot
        num_classes = len(np.unique(y))
        y_cat = np.zeros((y.shape[0], num_classes))
        y_cat[np.arange(y.shape[0]), y] = 1

        # التنبؤ والتقييم
        y_pred_prob = model.predict(X)
        y_pred = np.argmax(y_pred_prob, axis=1)

        accuracy = accuracy_score(y, y_pred)
        print(f"\n✅ Model accuracy on test set: {accuracy:.3f}\n")
        print("📊 Classification Report:")
        print(classification_report(y, y_pred, zero_division=0))

    else:
        print("ℹ️ Mode not recognized. Use 'analyze', 'train', or 'test'.")

if __name__ == "__main__":
    mode_arg = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    main(mode_arg)



# from app.services.data_loader import DataLoader
# from app.services.features import FeatureEngineer
# from app.services.training_pipeline import TrainingPipeline
# from app.core.database import SessionLocal
# import numpy as np
# import pprint

# def main(mode="train"):
#     """
#     mode: "train" لتدريب النموذج
#           "test" لتقييم النموذج على بيانات الاختبار
#     """
#     db = SessionLocal()
#     loader = DataLoader(db)
#     features_extractor = FeatureEngineer()

#     # 1️⃣ تحميل البيانات
#     characters = ["ا", "ب", "ت"]  # مثال لتجربة
#     data = loader.load_gestures_data(characters, limit_per_char=50)
#     print(f"✅ Loaded {len(data)} gestures\n")

#     # 2️⃣ استخراج الميزات
#     X, y = features_extractor.extract_features(data)
#     print(f"🔹 Feature dimensions: X={X.shape}, y={y.shape}")

#     # 3️⃣ اختيار الوضع
#     if mode == "train":
#         print("\n🎯 Starting training pipeline...")
#         pipeline = TrainingPipeline(db)
#         result = pipeline.train_model(characters)
#         print(f"Training completed. Test accuracy: {result['test_accuracy']:.3f}")
    
#     elif mode == "test":
#         print("\n🧪 Running testing on saved model...")
#         from tensorflow.keras.models import load_model
#         from tensorflow.keras.utils import to_categorical
#         from sklearn.metrics import accuracy_score, classification_report

#         model = load_model("arabic_gesture_lstm_final.h5")
#         num_classes = len(np.unique(y))
#         y_cat = to_categorical(y, num_classes)

#         y_pred_prob = model.predict(X)
#         y_pred = np.argmax(y_pred_prob, axis=1)

#         acc = accuracy_score(y, y_pred)
#         print(f"✅ Test accuracy: {acc:.3f}")

#         print("\n📊 Classification report:")
#         print(classification_report(y, y_pred, zero_division=0))

#     else:
#         print("❌ Invalid mode. Use 'train' or 'test'.")

# if __name__ == "__main__":
#     import sys
#     mode = sys.argv[1] if len(sys.argv) > 1 else "train"
#     main(mode)

