import numpy as np
import pickle
from app.services.features import FeatureEngineer
from tensorflow.keras.models import load_model

# تحميل الموديل
model = load_model("arabic_gesture_lstm_final.h5")
labels = ['ا', 'ب']
CONFIDENCE_THRESHOLD = 0.5

# تحميل الـ scaler اللي دربته
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# دالة جديدة لتحويل gesture للتنبؤ
def preprocess_gesture(gesture, max_timesteps=100):
    frames = gesture['frames']
    if not frames:
        return None

    sequence = []
    for frame in frames:
        if not frame['points']:
            continue
        x_coords = [p['x'] for p in frame['points']]
        y_coords = [p['y'] for p in frame['points']]
        pressure = [p['pressure'] for p in frame['points']]
        frame_features = [
            np.mean(x_coords), np.std(x_coords),
            np.mean(y_coords), np.std(y_coords),
            np.mean(pressure), np.std(pressure),
            len(frame['points'])
        ]
        sequence.append(frame_features)

    # padding أو truncation
    if len(sequence) < max_timesteps:
        pad_len = max_timesteps - len(sequence)
        sequence.extend([[0]*len(sequence[0])] * pad_len)
    else:
        sequence = sequence[:max_timesteps]

    X = np.array(sequence)
    # تطبيع باستخدام scaler من التدريب
    X_scaled = scaler.transform(X)
    return X_scaled.reshape(1, max_timesteps, X_scaled.shape[1])

def predict_gesture(gesture_data):
    X = preprocess_gesture(gesture_data)
    if X is None:
        return {"predicted_letter": 0}

    prediction = model.predict(X)[0]
    predicted_class = int(np.argmax(prediction))

    if prediction[predicted_class] < CONFIDENCE_THRESHOLD:
        return {"predicted_letter": 0}

    return {"predicted_letter": labels[predicted_class]}
