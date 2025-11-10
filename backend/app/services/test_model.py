# app/test_model.py
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from app.services.data_loader import DataLoader
from app.services.features import FeatureEngineer

def main():
    # 1️⃣ تحميل البيانات من API (نفس طريقة التدريب)
    loader = DataLoader(api_url="https://api.sydev.site/api/gestures")
    features_extractor = FeatureEngineer(max_timesteps=200)

    characters = ["ا", "ب", "ت"]
    data = loader.load_gestures_data(characters, limit_per_char=50)
    print(f"\n✅ Loaded {len(data)} gestures for testing\n")

    # 2️⃣ استخراج الميزات
    X, y = features_extractor.extract_features(data)
    print(f"🔹 Feature shape: {X.shape}, Number of classes: {len(np.unique(y))}")

    # 3️⃣ تحميل النموذج المدرب والـ Scaler و LabelEncoder
    model = load_model("arabic_gesture_cnn_final.h5")
    with open("scaler.pkl", "rb") as f:
        features_extractor.scaler = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        features_extractor.label_encoder = pickle.load(f)

    # 4️⃣ تحويل التسميات إلى one-hot
    num_classes = len(np.unique(y))
    y_cat = np.zeros((y.shape[0], num_classes))
    y_cat[np.arange(y.shape[0]), y] = 1  # one-hot manual

    # 5️⃣ التنبؤ
    y_pred_prob = model.predict(X)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # 6️⃣ تقييم النموذج
    from sklearn.metrics import accuracy_score, classification_report

    accuracy = accuracy_score(y, y_pred)
    print(f"\n✅ Model accuracy on test set: {accuracy:.3f}\n")

    print("📊 Classification Report:")
    print(classification_report(y, y_pred, zero_division=0))

if __name__ == "__main__":
    main()

