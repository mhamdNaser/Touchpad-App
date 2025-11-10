from app.services.data_loader import DataLoader
from app.services.features import FeatureEngineer
from app.core.database import SessionLocal
import pandas as pd
import numpy as np

def main():
    """
    🔍 تحليل الفيتشر فقط:
    - تحميل بيانات الإيماءات من قاعدة البيانات
    - استخراج الميزات باستخدام FeatureEngineer
    - عرض جدول بالميزات المجمعة لكل حرف
    """
    db = SessionLocal()
    loader = DataLoader(db)
    features_extractor = FeatureEngineer()

    # 1️⃣ تحميل البيانات
    characters = ["ا", "ب", "ت"]  # مثال لأحرف التجربة
    data = loader.load_gestures_data(characters, limit_per_char=50)
    print(f"✅ Loaded {len(data)} gestures\n")

    # 2️⃣ تحليل واستخراج الفيتشر
    aggregated_features = features_extractor.aggregate_by_character(data)

    # 3️⃣ طباعة الجدول التحليلي
    df = features_extractor.show_feature_table(aggregated_features)

    # 4️⃣ حفظ الجدول في ملف CSV لتصفحه لاحقًا
    df.to_csv("gesture_features_analysis.csv", encoding="utf-8-sig")
    print("💾 تم حفظ نتائج التحليل في: gesture_features_analysis.csv\n")


if __name__ == "__main__":
    main()



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

