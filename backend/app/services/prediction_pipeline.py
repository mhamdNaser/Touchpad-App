import os
import pickle
import numpy as np
from typing import Dict, Any

from tensorflow.keras.models import load_model
from app.services.advanced_feature_extractor import AdvancedFeatureExtractor
from app.services.normalization_utils import (
    normalize_positions,
    resample_frames,
    resample_points_per_frame
)


class PredictionPipeline:
    """
    Prediction pipeline identical to training:
    - Uses AdvancedFeatureExtractor
    - No scaler
    - Loads final keras model + label encoder
    """

    def __init__(self,
                 model_path: str = "arabic_gesture_cnn_final.keras",
                 label_encoder_path: str = "label_encoder.pkl",
                 max_timesteps: int = 200,
                 verbose: bool = True):

        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.max_timesteps = max_timesteps
        self.verbose = verbose

        # Extractor identical to training
        self.feature_extractor = AdvancedFeatureExtractor(
            max_timesteps=max_timesteps,
            verbose=False
        )

        # load model + labels
        self.model = self._load_model(self.model_path)
        self.label_encoder = self._load_pickle(self.label_encoder_path, required=True)

    # ---------------------------------------------
    # helpers
    # ---------------------------------------------
    def _load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        model = load_model(path)
        if self.verbose:
            print(f"✅ Loaded model from {path}")
        return model

    def _load_pickle(self, path: str, required: bool = True):
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Required file not found: {path}")
            return None

        with open(path, "rb") as f:
            obj = pickle.load(f)

        if self.verbose:
            print(f"✅ Loaded pickle: {path}")

        return obj

    # ---------------------------------------------
    # convert frontend data → training gesture format
    # ---------------------------------------------
    def _convert_frontend_to_training_format(self, gesture_from_frontend: Dict[str, Any]) -> Dict[str, Any]:

        frames_converted = []
        for frame in gesture_from_frontend.get("frames", []):
            frames_converted.append({
                "frame_id": frame.get("frame_id"),
                "timestamp": frame.get("ts") or frame.get("timestamp"),
                "delta_ms": frame.get("delta_ms"),
                "points": frame.get("points", []),
                "raw_payload": frame
            })

        return {
            "id": gesture_from_frontend.get("id", 0),
            "character": None,
            "start_time": gesture_from_frontend.get("start_time"),
            "end_time": gesture_from_frontend.get("end_time"),
            "duration_ms": gesture_from_frontend.get("duration_ms"),
            "frame_count": len(frames_converted),
            "frames": frames_converted
        }

    # ---------------------------------------------
    # MAIN: prediction
    # ---------------------------------------------
    def predict_gesture(self, gesture_from_frontend: Dict[str, Any], return_topk: int = 5) -> Dict[str, Any]:

        if not gesture_from_frontend.get("frames"):
            raise ValueError("❌ Empty gesture: no frames received.")

        # 1️⃣ تحويل البيانات إلى شكل التدريب
        gesture_ready = self._convert_frontend_to_training_format(gesture_from_frontend)

        # 2️⃣ تطبيق normalization و resampling على gesture بالكامل
        gesture_ready["frames"] = normalize_positions(gesture_ready["frames"])
        gesture_ready["frames"] = resample_frames(gesture_ready["frames"], target_frames=self.max_timesteps)
        gesture_ready["frames"] = resample_points_per_frame(gesture_ready["frames"], target_points=20)

        # 3️⃣ استخراج المميزات كما في التدريب
        seq = self.feature_extractor._gesture_to_sequence(gesture_ready)

        if seq is None or seq.size == 0:
            raise ValueError("❌ Failed to extract features from gesture.")

        feature_dim = seq.shape[1]

        # 4️⃣ Padding إذا كانت قصيرة
        if seq.shape[0] < self.max_timesteps:
            pad_length = self.max_timesteps - seq.shape[0]
            seq = np.vstack([seq, np.zeros((pad_length, feature_dim))])

        seq = np.nan_to_num(seq, nan=0.0)

        if seq.shape != (self.max_timesteps, feature_dim):
            raise RuntimeError(
                f"Shape mismatch: got {seq.shape}, expected {(self.max_timesteps, feature_dim)}"
            )

        # 5️⃣ تحضير النموذج
        X_input = seq[np.newaxis, :, :]  # (1, T, F)

        if self.verbose:
            print(f"📐 Input shape = {X_input.shape}")

        # 6️⃣ التنبؤ
        preds = self.model.predict(X_input, verbose=0)[0]

        pred_idx = int(np.argmax(preds))
        confidence = float(preds[pred_idx])

        # 7️⃣ فك التشفير إلى حرف
        try:
            predicted_char = self.label_encoder.inverse_transform([pred_idx])[0]
        except Exception:
            predicted_char = str(pred_idx)

        # top-k
        top_k = min(return_topk, len(preds))
        top_indices = preds.argsort()[::-1][:top_k]

        top = [
            {
                "index": int(i),
                "label": self.label_encoder.inverse_transform([int(i)])[0],
                "probability": float(preds[i])
            }
            for i in top_indices
        ]

        return {
            "predicted_index": pred_idx,
            "predicted_letter": predicted_char,
            "confidence": confidence,
            "probabilities": preds.tolist(),
            "top": top,
            "timesteps": self.max_timesteps,
            "num_features": feature_dim
        }