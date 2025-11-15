# import os
# import pickle
# import numpy as np
# from typing import Dict, Any

# from tensorflow.keras.models import load_model
# from app.services.features_visualizer import ProductionFeatureExtractor


# class PredictionPipeline:
#     """
#     Prediction pipeline compatible with the training pipeline:
#     - Uses ProductionFeatureExtractor to get the same per-frame features.
#     - Loads keras model and label_encoder (sklearn LabelEncoder).
#     - Returns predicted label for frontend.
#     """

#     def __init__(self,
#                  model_path: str = "arabic_gesture_cnn_final.h5",
#                  label_encoder_path: str = "label_encoder.pkl",
#                  max_timesteps: int = 150,
#                  verbose: bool = True):
#         self.model_path = model_path
#         self.label_encoder_path = label_encoder_path
#         self.max_timesteps = max_timesteps
#         self.verbose = verbose

#         # feature extractor (must match training)
#         self.feature_extractor = ProductionFeatureExtractor(max_timesteps=self.max_timesteps, verbose=False)

#         # load model and encoder
#         self.model = self._load_model(self.model_path)
#         self.label_encoder = self._load_pickle(self.label_encoder_path, required=True)

#     # ---------------- loading helpers ----------------
#     def _load_model(self, path: str):
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"Model file not found: {path}")
#         try:
#             model = load_model(path)
#             if self.verbose:
#                 print(f"✅ Loaded model from {path}")
#             return model
#         except Exception as e:
#             raise RuntimeError(f"Failed to load model from {path}: {e}")

#     def _load_pickle(self, path: str, required: bool = True):
#         if not os.path.exists(path):
#             if required:
#                 raise FileNotFoundError(f"Required file not found: {path}")
#             else:
#                 return None
#         try:
#             with open(path, "rb") as f:
#                 obj = pickle.load(f)
#             if self.verbose:
#                 print(f"✅ Loaded pickle: {path}")
#             return obj
#         except Exception as e:
#             if required:
#                 raise RuntimeError(f"Failed to load pickle {path}: {e}")
#             else:
#                 if self.verbose:
#                     print(f"⚠️ Could not load optional pickle {path}: {e}")
#                 return None

#     # ---------------- convert frontend payload -> training-like gesture dict ----------------
#     def _convert_frontend_to_training_format(self, gesture_from_frontend: Dict[str, Any]) -> Dict[str, Any]:
#         frames_converted = []
#         for frame in gesture_from_frontend.get("frames", []):
#             frames_converted.append({
#                 "frame_id": frame.get("frame_id"),
#                 "timestamp": frame.get("ts") or frame.get("timestamp"),
#                 "delta_ms": frame.get("delta_ms", None),
#                 "points": frame.get("points", []),
#                 "raw_payload": frame
#             })

#         return {
#             "id": gesture_from_frontend.get("id", 0),
#             "character": None,
#             "start_time": gesture_from_frontend.get("start_time"),
#             "end_time": gesture_from_frontend.get("end_time"),
#             "duration_ms": gesture_from_frontend.get("duration_ms", 0),
#             "frame_count": len(frames_converted),
#             "frames": frames_converted
#         }

#     # ---------------- helper: resize sequence to max_timesteps ----------------
#     def _resize_sequence(self, seq: np.ndarray) -> np.ndarray:
#         T, F = seq.shape
#         if T == self.max_timesteps:
#             return seq
#         elif T < 2:
#             repeats = np.ceil(self.max_timesteps / T).astype(int)
#             seq_resized = np.tile(seq, (repeats, 1))[:self.max_timesteps]
#             return seq_resized
#         else:
#             x_old = np.linspace(0, 1, T)
#             x_new = np.linspace(0, 1, self.max_timesteps)
#             seq_resized = np.zeros((self.max_timesteps, F), dtype=seq.dtype)
#             for f in range(F):
#                 seq_resized[:, f] = np.interp(x_new, x_old, seq[:, f])
#             return seq_resized

#     # ---------------- main predict function ----------------
#     def predict_gesture(self, gesture_from_frontend: Dict[str, Any], return_topk: int = 5) -> Dict[str, Any]:
#         # 1) convert format
#         gesture_ready = self._convert_frontend_to_training_format(gesture_from_frontend)

#         # 2) extract sequence using the exact extractor used in training
#         seq = self.feature_extractor._gesture_to_sequence(gesture_ready)

#         if seq is None or seq.size == 0:
#             raise ValueError("Failed to extract features from gesture (no valid frames).")

#         # 2.5) resize short sequences to max_timesteps
#         seq = self._resize_sequence(seq)
#         if self.verbose:
#             print(f"🔄 Resized sequence shape: {seq.shape}")

#         # 3) guard against NaN / inf
#         seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

#         # 4) use raw features directly (no scaler)
#         X_input = np.expand_dims(seq, axis=0)
#         if self.verbose:
#             print("ℹ️ Using raw features without scaler.")

#         # 5) predict
#         preds = self.model.predict(X_input, verbose=0)
#         probs = np.asarray(preds[0] if preds.ndim == 2 else preds, dtype=np.float32)
#         probs /= probs.sum()  # normalize

#         pred_idx = int(np.argmax(probs))
#         confidence = float(probs[pred_idx])

#         # 6) invert label encoding
#         try:
#             predicted_char = self.label_encoder.inverse_transform([pred_idx])[0]
#         except Exception:
#             predicted_char = str(pred_idx)

#         # 7) top-k probabilities
#         top_k = min(return_topk, len(probs))
#         top_indices = (-probs).argsort()[:top_k]
#         top = [{"index": int(i),
#                 "label": (self.label_encoder.inverse_transform([int(i)])[0] if hasattr(self.label_encoder, "inverse_transform") else str(i)),
#                 "probability": float(probs[int(i)])} for i in top_indices]

#         return {
#             "predicted_index": pred_idx,
#             "predicted_letter": predicted_char,
#             "confidence": confidence,
#             "probabilities": probs.tolist(),
#             "top": top,
#             "timesteps": int(seq.shape[0]),
#             "num_features": int(seq.shape[1])
#         }

import os
import pickle
import numpy as np
from typing import Dict, Any

from tensorflow.keras.models import load_model
from app.services.features_visualizer import ProductionFeatureExtractor


class PredictionPipeline:
    """
    Prediction pipeline compatible with the training pipeline (old features):
    - Uses ProductionFeatureExtractor to get the same per-frame features (x, y, pressure, angle, delta_s).
    - No scaler applied.
    - Loads keras model and label_encoder.
    - Returns predicted label for frontend.
    """

    def __init__(self,
                 model_path: str = "arabic_gesture_cnn_final.h5",
                 label_encoder_path: str = "label_encoder.pkl",
                 max_timesteps: int = 50,
                 verbose: bool = True):
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.max_timesteps = max_timesteps
        self.verbose = verbose

        # feature extractor (old features only)
        self.feature_extractor = ProductionFeatureExtractor(
            max_timesteps=self.max_timesteps,
            verbose=False
        )

        # load model and encoder
        self.model = self._load_model(self.model_path)
        self.label_encoder = self._load_pickle(self.label_encoder_path, required=True)

    # ---------------- loading helpers ----------------
    def _load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        try:
            model = load_model(path)
            if self.verbose:
                print(f"✅ Loaded model from {path}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {e}")

    def _load_pickle(self, path: str, required: bool = True):
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Required file not found: {path}")
            else:
                return None
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if self.verbose:
                print(f"✅ Loaded pickle: {path}")
            return obj
        except Exception as e:
            if required:
                raise RuntimeError(f"Failed to load pickle {path}: {e}")
            else:
                if self.verbose:
                    print(f"⚠️ Could not load optional pickle {path}: {e}")
                return None

    # ---------------- convert frontend payload -> training-like gesture dict ----------------
    def _convert_frontend_to_training_format(self, gesture_from_frontend: Dict[str, Any]) -> Dict[str, Any]:
        frames_converted = []
        for frame in gesture_from_frontend.get("frames", []):
            frames_converted.append({
                "frame_id": frame.get("frame_id"),
                "timestamp": frame.get("ts") or frame.get("timestamp"),
                "delta_ms": frame.get("delta_ms", None),
                "points": frame.get("points", []),
                "raw_payload": frame
            })

        return {
            "id": gesture_from_frontend.get("id", 0),
            "character": None,
            "start_time": gesture_from_frontend.get("start_time"),
            "end_time": gesture_from_frontend.get("end_time"),
            "duration_ms": gesture_from_frontend.get("duration_ms", 0),
            "frame_count": len(frames_converted),
            "frames": frames_converted
        }

    # ---------------- helper: resize sequence to max_timesteps ----------------
    def _resize_sequence(self, seq: np.ndarray) -> np.ndarray:
        T, F = seq.shape
        if T == self.max_timesteps:
            return seq
        elif T < 2:
            repeats = np.ceil(self.max_timesteps / T).astype(int)
            seq_resized = np.tile(seq, (repeats, 1))[:self.max_timesteps]
            return seq_resized
        else:
            x_old = np.linspace(0, 1, T)
            x_new = np.linspace(0, 1, self.max_timesteps)
            seq_resized = np.zeros((self.max_timesteps, F), dtype=seq.dtype)
            for f in range(F):
                seq_resized[:, f] = np.interp(x_new, x_old, seq[:, f])
            return seq_resized

    # ---------------- main predict function ----------------
    def predict_gesture(self, gesture_from_frontend: Dict[str, Any], return_topk: int = 5) -> Dict[str, Any]:
        # 1) convert format
        gesture_ready = self._convert_frontend_to_training_format(gesture_from_frontend)

        # 2) extract sequence using the exact extractor used in training
        seq = self.feature_extractor._gesture_to_sequence(gesture_ready)

        if seq is None or seq.size == 0:
            raise ValueError("Failed to extract features from gesture (no valid frames).")

        # 2.5) resize short sequences to max_timesteps
        seq = self._resize_sequence(seq)
        if self.verbose:
            print(f"🔄 Resized sequence shape: {seq.shape}")

        # 3) guard against NaN / inf
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        # 4) use raw features directly (no scaler)
        X_input = np.expand_dims(seq, axis=0)
        if self.verbose:
            print("ℹ️ Using raw features without scaler.")

        # 5) predict
        preds = self.model.predict(X_input, verbose=0)
        probs = np.asarray(preds[0] if preds.ndim == 2 else preds, dtype=np.float32)
        probs /= probs.sum()  # normalize

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        # 6) invert label encoding
        try:
            predicted_char = self.label_encoder.inverse_transform([pred_idx])[0]
        except Exception:
            predicted_char = str(pred_idx)

        # 7) top-k probabilities
        top_k = min(return_topk, len(probs))
        top_indices = (-probs).argsort()[:top_k]
        top = [{"index": int(i),
                "label": (self.label_encoder.inverse_transform([int(i)])[0] if hasattr(self.label_encoder, "inverse_transform") else str(i)),
                "probability": float(probs[int(i)])} for i in top_indices]

        return {
            "predicted_index": pred_idx,
            "predicted_letter": predicted_char,
            "confidence": confidence,
            "probabilities": probs.tolist(),
            "top": top,
            "timesteps": int(seq.shape[0]),
            "num_features": int(seq.shape[1])
        }
