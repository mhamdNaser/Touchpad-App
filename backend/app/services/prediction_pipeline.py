# app/services/prediction_pipeline.py
import os
import pickle
import numpy as np
from typing import Dict, Any
from tensorflow.keras.models import load_model
from app.services.gesture_data_loader import GestureDataLoader
from app.services.advanced_feature_extractor import AdvancedFeatureExtractor


class PredictionPipeline:
    """
    Prediction pipeline aligned with training:
    - Uses GestureDataLoader to process frontend gestures
    - Uses AdvancedFeatureExtractor for feature extraction
    - Loads trained Keras model + label encoder
    """
    def __init__(
        self,
        model_path: str = "ai_model/arabic_gesture_cnn_lstm_final.keras",
        label_encoder_path: str = "ai_model/label_encoder.pkl",
        max_timesteps: int = 200,
        max_points: int = 20,
        verbose: bool = True
    ):
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.max_timesteps = max_timesteps
        self.max_points = max_points
        self.verbose = verbose

        # Data loader identical to training
        self.data_loader = GestureDataLoader(
            target_frames=max_timesteps,
            target_points=max_points
        )

        # Feature extractor identical to training
        self.feature_extractor = AdvancedFeatureExtractor(
            max_timesteps=max_timesteps,
            verbose=False
        )

        # Load model + label encoder
        self.model = self._load_model(self.model_path)
        self.label_encoder = self._load_pickle(self.label_encoder_path, required=True)

    # ---------------------------
    # helpers
    # ---------------------------
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

    # ---------------------------
    # convert frontend → training-like gesture
    # ---------------------------
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

    # ---------------------------
    # MAIN: predict gesture
    # ---------------------------
    def predict_gesture(self, gesture_from_frontend: Dict[str, Any], return_topk: int = 5) -> Dict[str, Any]:
        if not gesture_from_frontend.get("frames"):
            raise ValueError("❌ Empty gesture: no frames received.")

        # 1) Convert frontend gesture → training-like format
        gesture_ready = self._convert_frontend_to_training_format(gesture_from_frontend)

        # 2) Extract features using the same extractor as training
        seq = self.feature_extractor._gesture_to_sequence(gesture_ready)
        if seq is None or seq.size == 0:
            raise ValueError("❌ Failed to extract features from gesture.")

        feature_dim = seq.shape[1]

        # 3) Pad sequence إذا كانت قصيرة
        if seq.shape[0] < self.max_timesteps:
            pad_length = self.max_timesteps - seq.shape[0]
            seq = np.vstack([seq, np.zeros((pad_length, feature_dim))])

        # 4) Clean NaN values
        seq = np.nan_to_num(seq, nan=0.0)

        # تحقق من الشكل
        if seq.shape != (self.max_timesteps, feature_dim):
            raise RuntimeError(f"Shape mismatch: got {seq.shape}, expected {(self.max_timesteps, feature_dim)}")

        # 5) Prepare input for model
        X_input = seq[np.newaxis, :, :]  # (1, T, F)
        if self.verbose:
            print(f"📐 Input shape = {X_input.shape}")

        # 6) Run model
        preds = self.model.predict(X_input, verbose=0)[0]

        pred_idx = int(np.argmax(preds))
        confidence = float(preds[pred_idx])

        # 7) Decode label
        try:
            predicted_char = self.label_encoder.inverse_transform([pred_idx])[0]
        except Exception:
            predicted_char = str(pred_idx)

        # 8) Top-k predictions
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
