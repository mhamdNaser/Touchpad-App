import pickle
import numpy as np
from tensorflow.keras.models import load_model
from app.services.advanced_feature_extractor import FeatureEngineer
from app.services.preprocess import Preprocessor


class PredictionPipeline:
    """
    نسخة مطورة من واجهة TrainingPipeline، مخصصة للتنبؤ بالإيماءات القادمة من الفرونت إند.
    تشمل:
    - تحويل بيانات الفرونت إند لتشبه بيانات التدريب
    - استخراج الميزات
    - التطبيع باستخدام scaler المحفوظ
    - التنبؤ بالحرف باستخدام النموذج والـ label_encoder
    """

    def __init__(self,
                 model_path="arabic_gesture_cnn_final.h5",
                 scaler_path="scaler.pkl",
                 label_encoder_path="label_encoder.pkl",
                 max_timesteps: int = 100):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.label_encoder_path = label_encoder_path
        self.max_timesteps = max_timesteps

        # تحميل المكونات
        self.feature_engineer = FeatureEngineer(max_timesteps=self.max_timesteps)
        self.model = self._load_model()
        self.scaler = self._load_pickle(self.scaler_path)
        self.label_encoder = self._load_pickle(self.label_encoder_path)
        self.preprocessor = Preprocessor()

    # ================== أدوات التحميل ==================
    def _load_model(self):
        try:
            model = load_model(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model: {e}")

    def _load_pickle(self, path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load pickle file {path}: {e}")

    # ================== تحويل بيانات الفرونت إند ==================
    def _convert_frontend_to_training_format(self, gesture_from_frontend: dict) -> dict:
        """
        تحويل بيانات الإيماءة القادمة من الفرونت إند إلى نفس شكل بيانات التدريب.
        """
        frames_converted = []
        for frame in gesture_from_frontend.get("frames", []):
            frames_converted.append({
                "frame_id": frame.get("frame_id"),
                "timestamp": frame.get("ts"),
                "points_count": len(frame.get("points", [])),
                "raw_payload": {
                    "ts": frame.get("ts"),
                    "frame_id": frame.get("frame_id"),
                    "points": frame.get("points", [])
                },
                "points": frame.get("points", [])
            })

        return {
            "id": 0,
            "character": None,
            "start_time": gesture_from_frontend.get("start_time"),
            "end_time": gesture_from_frontend.get("end_time"),
            "duration_ms": gesture_from_frontend.get("duration_ms", 0),
            "frame_count": len(frames_converted),
            "frames": frames_converted
        }

    # ================== دالة التنبؤ الرئيسية ==================
    def predict_gesture(self, gesture_from_frontend: dict):
        """
        يستقبل بيانات من الفرونت إند ويعيد التنبؤ بالحرف مع نسبة الثقة.
        """
        # 1️⃣ تحويل تنسيق البيانات
        gesture_ready = self._convert_frontend_to_training_format(gesture_from_frontend)

        # 2️⃣ تمريرها عبر Preprocessor
        processed_gesture = self.preprocessor.process_gesture(gesture_ready)
        if processed_gesture is None:
            raise ValueError("❌ No valid frames found in gesture.")

        # 3️⃣ استخراج الميزات
        seq = self.feature_engineer.extract_sequence_features(processed_gesture)
        if seq is None:
            raise ValueError("❌ Failed to extract sequence features.")

        T, F = seq.shape  # timesteps, features

        # 4️⃣ معالجة الحالات القصيرة (عدد فريمات قليل جدًا)
        if T < 3:
            while seq.shape[0] < 3:
                seq = np.concatenate([seq, seq[-1:, :]], axis=0)
            T, F = seq.shape

        # 5️⃣ تطبيع بنفس الـ scaler المستخدم أثناء التدريب
        # X_flat = seq.reshape(1, -1)
        # X_scaled = self.scaler.transform(X_flat)
        # X_input = X_scaled.reshape(1, T, F)
        X_scaled = np.array([self.scaler.transform(seq)])  # الشكل (1, timesteps, features)
        X_input = X_scaled

        # 6️⃣ التنبؤ باستخدام النموذج
        preds = self.model.predict(X_input)[0]
        pred_idx = int(np.argmax(preds))
        confidence = float(preds[pred_idx])

        # 7️⃣ تحويل التنبؤ إلى الحرف المقابل
        predicted_char = self.label_encoder.inverse_transform([pred_idx])[0]

        return {
            "predicted_letter": predicted_char,
            "confidence": confidence,
            "timesteps": int(T),
            "num_features": int(F)
        }
