# app/services/prediction_pipeline.py

import os
import pickle
import numpy as np
from typing import Dict, Any, List, Optional
from tensorflow.keras.models import load_model

from app.services.gesture_data_loader import GestureDataLoader
from app.services.advanced_feature_extractor import AdvancedFeatureExtractor


class PredictionPipeline:
    """
    Enhanced prediction pipeline aligned with training:
    - Uses GestureDataLoader to process frontend gestures
    - Uses AdvancedFeatureExtractor for FULL feature extraction
    - Loads trained Keras model + label encoder
    - Added error handling and quality checks
    """
    def __init__(
        self,
        model_path: str = "ai_model/final_model.keras",
        label_encoder_path: str = "ai_model/label_encoder.pkl",
        max_timesteps: int = 200,
        verbose: bool = True
    ):
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.max_timesteps = max_timesteps
        self.verbose = verbose

        # Data loader identical to training
        self.data_loader = GestureDataLoader()

        # Feature extractor identical to training
        self.feature_extractor = AdvancedFeatureExtractor(
            max_timesteps=max_timesteps,
            verbose=False
        )

        # Load model + label encoder
        self.model = self._load_model(self.model_path)
        self.label_encoder = self._load_pickle(self.label_encoder_path, required=True)
        
        # Get expected feature dimension
        self.expected_feature_dim = self.feature_extractor.get_feature_dimension()

    # ---------------------------------------
    # Loading helpers
    # ---------------------------------------
    def _load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        try:
            model = load_model(path)
            if self.verbose:
                print(f"✅ Loaded model from {path}")
                print(f"📐 Model input shape: {model.input_shape}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _load_pickle(self, path: str, required: bool = True):
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Required file not found: {path}")
            return None
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if self.verbose:
                print(f"✅ Loaded pickle: {path}")
                if hasattr(obj, 'classes_'):
                    print(f"🎯 Label classes: {len(obj.classes_)}")
            return obj
        except Exception as e:
            raise RuntimeError(f"Failed to load pickle: {e}")

    # ---------------------------------------
    # Gesture Quality Validation
    # ---------------------------------------
    def _validate_gesture_quality(self, gesture: Dict) -> Dict[str, Any]:
        """التحقق من جودة الإيماءة قبل التنبؤ"""
        frames = gesture.get("frames", [])
        
        quality_report = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check frame count
        if len(frames) < 3:
            quality_report["is_valid"] = False
            quality_report["errors"].append(f"Too few frames: {len(frames)} (minimum: 3)")
        
        # Check total points
        total_points = sum(len(frame.get("points", [])) for frame in frames)
        if total_points < 5:
            quality_report["is_valid"] = False
            quality_report["errors"].append(f"Too few points: {total_points} (minimum: 5)")
        
        # Check frame timing
        timestamps = [f.get("timestamp", 0) for f in frames if f.get("timestamp")]
        if len(timestamps) > 1:
            duration = max(timestamps) - min(timestamps)
            if duration < 10:  # Very short gesture (ms)
                quality_report["warnings"].append(f"Very short gesture duration: {duration}ms")
        
        # Check point distribution
        all_points = []
        for frame in frames:
            all_points.extend(frame.get("points", []))
        
        if all_points:
            xs = [p.get("x", 0) for p in all_points]
            ys = [p.get("y", 0) for p in all_points]
            
            if max(xs) - min(xs) < 0.01:  # Very small movement in X
                quality_report["warnings"].append("Very small movement in X direction")
            if max(ys) - min(ys) < 0.01:  # Very small movement in Y
                quality_report["warnings"].append("Very small movement in Y direction")
        
        return quality_report

    # ---------------------------------------
    # Convert frontend → training-like format
    # ---------------------------------------
    def _convert_frontend_to_training_format(self, gesture_from_frontend: Dict[str, Any]) -> Dict[str, Any]:
        """تحويل بيانات الواجهة الأمامية إلى تنسيق التدريب"""
        frames_converted = []

        for frame in gesture_from_frontend.get("frames", []):
            # Ensure all points have required fields
            processed_points = []
            for point in frame.get("points", []):
                processed_point = {
                    "x": point.get("x", 0.0),
                    "y": point.get("y", 0.0),
                    "pressure": point.get("pressure", 1.0)
                }
                processed_points.append(processed_point)
            
            frames_converted.append({
                "frame_id": frame.get("frame_id"),
                "timestamp": frame.get("ts") or frame.get("timestamp") or 0,
                "delta_ms": frame.get("delta_ms") or 16,  # Default 16ms if missing
                "points": processed_points
            })

        return {
            "id": gesture_from_frontend.get("id", 0),
            "character": None,   # prediction will fill this
            "start_time": gesture_from_frontend.get("start_time"),
            "end_time": gesture_from_frontend.get("end_time"),
            "duration_ms": gesture_from_frontend.get("duration_ms"),
            "frame_count": len(frames_converted),
            "frames": frames_converted
        }

    # ---------------------------------------
    # Feature Processing & Validation
    # ---------------------------------------
    def _process_features(self, gesture: Dict) -> np.ndarray:
        """معالجة الميزات مع التحقق من الجودة"""
        # Extract features
        seq = self.feature_extractor.gesture_to_full_feature_vector(gesture)

        if seq is None or seq.size == 0:
            raise ValueError("❌ Failed to extract features from gesture")

        # Check feature dimension
        timesteps, feature_dim = seq.shape
        if feature_dim != self.expected_feature_dim:
            raise ValueError(f"❌ Feature dimension mismatch: expected {self.expected_feature_dim}, got {feature_dim}")

        # Pad/truncate to max_timesteps
        if timesteps < self.max_timesteps:
            padding = np.zeros((self.max_timesteps - timesteps, feature_dim))
            seq = np.vstack([seq, padding])
        else:
            seq = seq[:self.max_timesteps]

        # Handle NaN values
        seq = np.nan_to_num(seq, nan=0.0, posinf=1.0, neginf=-1.0)

        # Final shape validation
        if seq.shape != (self.max_timesteps, feature_dim):
            raise RuntimeError(f"❌ Shape mismatch: got {seq.shape}, expected {(self.max_timesteps, feature_dim)}")

        return seq

    # ---------------------------------------
    # Prediction with Confidence Scoring
    # ---------------------------------------
    def _calculate_confidence_score(self, probabilities: np.ndarray, top_prediction: float) -> float:
        """حساب درجة الثقة بناءً على توزيع الاحتمالات"""
        # Normal confidence (highest probability)
        confidence = top_prediction
        
        # Apply penalty if second-best is too close
        sorted_probs = np.sort(probabilities)[::-1]
        if len(sorted_probs) > 1:
            margin = sorted_probs[0] - sorted_probs[1]
            if margin < 0.3:  # If top two are close
                confidence *= 0.7  # Reduce confidence
        
        # Apply penalty for low maximum probability
        if top_prediction < 0.6:
            confidence *= 0.8
        
        return min(confidence, 1.0)  # Cap at 1.0

    # ---------------------------------------
    # MAIN PREDICTION
    # ---------------------------------------
    def predict_gesture(self, gesture_from_frontend: Dict[str, Any], return_topk: int = 5) -> Dict[str, Any]:
        """
        التنبؤ بالإيماءة مع معالجة محسنة للأخطاء
        """
        try:
            # Input validation
            if not gesture_from_frontend or not gesture_from_frontend.get("frames"):
                return {
                    "success": False,
                    "error": "Empty gesture: no frames received",
                    "predicted_letter": "?",
                    "confidence": 0.0,
                    "quality_report": {"is_valid": False, "errors": ["No frames provided"]}
                }

            # 1) Convert frontend format
            gesture_ready = self._convert_frontend_to_training_format(gesture_from_frontend)

            # 2) Quality check
            quality_report = self._validate_gesture_quality(gesture_ready)
            
            if not quality_report["is_valid"]:
                return {
                    "success": False,
                    "error": "Gesture quality check failed",
                    "predicted_letter": "?",
                    "confidence": 0.0,
                    "quality_report": quality_report
                }

            # 3) Process features
            seq = self._process_features(gesture_ready)

            # 4) Prepare for model
            X_input = seq[np.newaxis, :, :]  # (1, T, F)
            
            if self.verbose:
                print(f"📐 Input shape to model: {X_input.shape}")
                print(f"⚠️ Quality warnings: {quality_report['warnings']}")

            # 5) Predict
            preds = self.model.predict(X_input, verbose=0)[0]

            # 6) Process results
            pred_idx = int(np.argmax(preds))
            top_probability = float(preds[pred_idx])
            
            # Calculate enhanced confidence
            confidence = self._calculate_confidence_score(preds, top_probability)

            # 7) Decode label
            try:
                predicted_char = self.label_encoder.inverse_transform([pred_idx])[0]
            except Exception as e:
                predicted_char = f"Class_{pred_idx}"
                if self.verbose:
                    print(f"⚠️ Label decoding warning: {e}")

            # 8) Top-k predictions
            top_k = min(return_topk, len(preds))
            top_indices = preds.argsort()[::-1][:top_k]

            top_predictions = []
            for i in top_indices:
                try:
                    label = self.label_encoder.inverse_transform([int(i)])[0]
                except:
                    label = f"Class_{i}"
                
                top_predictions.append({
                    "index": int(i),
                    "label": label,
                    "probability": float(preds[i])
                })

            # 9) Return comprehensive result
            result = {
                "success": True,
                "predicted_index": pred_idx,
                "predicted_letter": predicted_char,
                "confidence": confidence,
                "raw_confidence": top_probability,  # Original probability
                "probabilities": preds.tolist(),
                "top": top_predictions,
                "timesteps": self.max_timesteps,
                "num_features": self.expected_feature_dim,
                "quality_report": quality_report,
                "feature_shape": X_input.shape
            }
            
            if self.verbose:
                print(f"🎯 Prediction: {predicted_char} (confidence: {confidence:.3f})")
                
            return result

        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "predicted_letter": "?",
                "confidence": 0.0,
                "quality_report": {"is_valid": False, "errors": [error_msg]}
            }

    # ---------------------------------------
    # Utility Methods
    # ---------------------------------------
    def get_model_info(self) -> Dict[str, Any]:
        """الحصول على معلومات النموذج"""
        return {
            "model_path": self.model_path,
            "input_shape": self.model.input_shape,
            "max_timesteps": self.max_timesteps,
            "feature_dimension": self.expected_feature_dim,
            "num_classes": len(self.label_encoder.classes_),
            "classes": self.label_encoder.classes_.tolist()
        }

    def get_feature_names(self) -> List[str]:
        """الحصول على أسماء الميزات المستخدمة"""
        return self.feature_extractor.get_feature_names()

    def batch_predict(self, gestures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """التنبؤ بعدة إيماءات دفعة واحدة"""
        return [self.predict_gesture(gesture) for gesture in gestures]


# Example usage and testing
if __name__ == "__main__":
    # Test the pipeline
    pipeline = PredictionPipeline(verbose=True)
    
    print("🔍 Model Info:")
    print(pipeline.get_model_info())
    
    print("📊 Feature Names:")
    print(pipeline.get_feature_names())