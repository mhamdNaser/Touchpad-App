import os
import pickle
import numpy as np
from typing import Dict, Any

from tensorflow.keras.models import load_model
from app.services.advanced_feature_extractor import AdvancedFeatureExtractor


class PredictionPipeline:
    """
    Prediction pipeline identical to training:
    - Uses AdvancedFeatureExtractor
    - No scaler
    - Loads final keras model + label encoder
    """

    def __init__(self,
                 model_path: str = "arabic_gesture_cnn_final.h5",
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

        # 1) convert to training-like format
        gesture_ready = self._convert_frontend_to_training_format(gesture_from_frontend)

        # 2) extract features exactly like training
        seq = self.feature_extractor._gesture_to_sequence(gesture_ready)

        if seq is None or seq.size == 0:
            raise ValueError("❌ Failed to extract features from gesture.")

        feature_dim = seq.shape[1]

        # --- padding إذا كانت قصيرة ---
        if seq.shape[0] < self.max_timesteps:
            pad_length = self.max_timesteps - seq.shape[0]
            seq = np.vstack([seq, np.zeros((pad_length, feature_dim))])

        # clean invalid values
        seq = np.nan_to_num(seq, nan=0.0)

        # تحقق من الشكل
        if seq.shape != (self.max_timesteps, feature_dim):
            raise RuntimeError(
                f"Shape mismatch: got {seq.shape}, expected {(self.max_timesteps, feature_dim)}"
            )

        # prepare for model input
        X_input = seq[np.newaxis, :, :]  # (1, T, F)

        if self.verbose:
            print(f"📐 Input shape = {X_input.shape}")

        # 3) run model
        preds = self.model.predict(X_input, verbose=0)[0]

        pred_idx = int(np.argmax(preds))
        confidence = float(preds[pred_idx])

        # 4) decode label
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

# import os
# import pickle
# import numpy as np
# from typing import Dict, Any

# from tensorflow.keras.models import load_model
# from app.services.advanced_feature_extractor import AdvancedFeatureExtractor


# class PredictionPipeline:
#     """
#     Prediction pipeline identical to training:
#     - Uses AdvancedFeatureExtractor
#     - No scaler
#     - Loads final keras model + label encoder
#     - Supports gestures containing one or multiple letters
#     """

#     def __init__(self,
#                  model_path: str = "arabic_gesture_cnn_final.h5",
#                  label_encoder_path: str = "label_encoder.pkl",
#                  max_timesteps: int = 200,
#                  verbose: bool = True):

#         self.model_path = model_path
#         self.label_encoder_path = label_encoder_path
#         self.max_timesteps = max_timesteps
#         self.verbose = verbose

#         # Extractor identical to training
#         self.feature_extractor = AdvancedFeatureExtractor(
#             max_timesteps=max_timesteps,
#             verbose=False
#         )

#         # load model + labels
#         self.model = self._load_model(self.model_path)
#         self.label_encoder = self._load_pickle(self.label_encoder_path, required=True)

#     # ---------------------------------------------
#     # helpers
#     # ---------------------------------------------
#     def _load_model(self, path: str):
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"Model file not found: {path}")

#         model = load_model(path)
#         if self.verbose:
#             print(f"✅ Loaded model from {path}")
#         return model

#     def _load_pickle(self, path: str, required: bool = True):
#         if not os.path.exists(path):
#             if required:
#                 raise FileNotFoundError(f"Required file not found: {path}")
#             return None

#         with open(path, "rb") as f:
#             obj = pickle.load(f)

#         if self.verbose:
#             print(f"✅ Loaded pickle: {path}")

#         return obj

#     # ---------------------------------------------
#     # convert frontend data → training gesture format
#     # ---------------------------------------------
#     def _convert_frontend_to_training_format(self, gesture_from_frontend: Dict[str, Any]) -> Dict[str, Any]:

#         frames_converted = []
#         for frame in gesture_from_frontend.get("frames", []):
#             frames_converted.append({
#                 "frame_id": frame.get("frame_id"),
#                 "timestamp": frame.get("ts") or frame.get("timestamp"),
#                 "delta_ms": frame.get("delta_ms"),
#                 "points": frame.get("points", []),
#                 "raw_payload": frame
#             })

#         return {
#             "id": gesture_from_frontend.get("id", 0),
#             "character": None,
#             "start_time": gesture_from_frontend.get("start_time"),
#             "end_time": gesture_from_frontend.get("end_time"),
#             "duration_ms": gesture_from_frontend.get("duration_ms"),
#             "frame_count": len(frames_converted),
#             "frames": frames_converted
#         }

#     # ---------------------------------------------
#     # MAIN: prediction
#     # ---------------------------------------------
#     def predict_gesture(self, gesture_from_frontend: Dict[str, Any], return_topk: int = 5) -> Dict[str, Any]:

#         if not gesture_from_frontend.get("frames"):
#             raise ValueError("❌ Empty gesture: no frames received.")

#         # 1) convert to training-like format
#         gesture_ready = self._convert_frontend_to_training_format(gesture_from_frontend)

#         # 2) extract features exactly like training
#         seq = self.feature_extractor._gesture_to_sequence(gesture_ready)

#         if seq is None or seq.size == 0:
#             raise ValueError("❌ Failed to extract features from gesture.")

#         feature_dim = seq.shape[1]
#         seq_length = seq.shape[0]

#         # --- sliding window parameters ---
#         window_size = self.max_timesteps
#         stride = max(1, window_size // 2)  # 50% overlap

#         predicted_sequence = []
#         confidence_sequence = []
#         probability_sequence = []

#         for start in range(0, seq_length, stride):
#             end = start + window_size
#             window_seq = seq[start:end]

#             # padding إذا كانت قصيرة
#             if window_seq.shape[0] < window_size:
#                 pad_length = window_size - window_seq.shape[0]
#                 window_seq = np.vstack([window_seq, np.zeros((pad_length, feature_dim))])

#             # clean invalid values
#             window_seq = np.nan_to_num(window_seq, nan=0.0)
#             X_input = window_seq[np.newaxis, :, :]  # (1, T, F)

#             if self.verbose:
#                 print(f"📐 Window input shape = {X_input.shape}")

#             # 3) run model
#             preds = self.model.predict(X_input, verbose=0)[0]

#             pred_idx = int(np.argmax(preds))
#             confidence = float(preds[pred_idx])
#             predicted_char = self.label_encoder.inverse_transform([pred_idx])[0]

#             # حفظ النتائج لكل نافذة
#             predicted_sequence.append(predicted_char)
#             confidence_sequence.append(confidence)
#             probability_sequence.append(preds.tolist())  # softmax لكل حرف

#         # --- دمج النتائج للتخلص من التكرار الناجم عن النوافذ المتداخلة ---
#         final_sequence = []
#         final_confidences = []
#         final_probabilities = []
#         prev_char = None

#         for c, conf, prob in zip(predicted_sequence, confidence_sequence, probability_sequence):
#             if c != prev_char:
#                 final_sequence.append(c)
#                 final_confidences.append(conf)
#                 final_probabilities.append(prob)
#                 prev_char = c

#         # --- top-k من آخر نافذة ---
#         last_preds = preds
#         top_k = min(return_topk, len(last_preds))
#         top_indices = last_preds.argsort()[::-1][:top_k]
#         top = [
#             {
#                 "index": int(i),
#                 "label": self.label_encoder.inverse_transform([int(i)])[0],
#                 "probability": float(last_preds[i])
#             }
#             for i in top_indices
#         ]

#         return {
#             "predicted_sequence": final_sequence,        # تسلسل الحروف
#             "confidences": final_confidences,           # ثقة كل حرف
#             "probabilities": final_probabilities,       # احتمالات كل حرف لكل نافذة
#             "top": top,                                 # top-k من آخر نافذة
#             "timesteps": self.max_timesteps,
#             "num_features": feature_dim
#         }
