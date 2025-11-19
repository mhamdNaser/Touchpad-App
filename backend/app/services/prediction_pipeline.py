# app/services/prediction_pipeline.py

import os
import pickle
import numpy as np
from typing import Dict, Any, List, Optional
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

from app.services.gesture_data_loader import GestureDataLoader
from app.services.advanced_feature_extractor import AdvancedFeatureExtractor


class PredictionPipeline:
    """
    Prediction pipeline aligned with TRAINING (Global Features)
    """
    def __init__(
        self,
        model_path: str = "ai_model/final_model.keras",
        label_encoder_path: str = "ai_model/label_encoder.pkl",
        scaler_path: str = "ai_model/scaler.pkl",
        verbose: bool = True
    ):
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.scaler_path = scaler_path
        self.verbose = verbose

        # Load all trained components
        self.model = self._load_model(self.model_path)
        self.label_encoder = self._load_pickle(self.label_encoder_path, required=True)
        self.scaler = self._load_pickle(self.scaler_path, required=True)
        
        # Feature extractor for GLOBAL features (same as training)
        self.feature_extractor = AdvancedFeatureExtractor(verbose=False)

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
                "timestamp": frame.get("ts") or frame.get("timestamp") or 0,
                "delta_ms": frame.get("delta_ms") or 16,
                "points": processed_points
            })

        return {
            "frames": frames_converted,
            "duration_ms": gesture_from_frontend.get("duration_ms", 0),
            "start_time": gesture_from_frontend.get("start_time"),
            "end_time": gesture_from_frontend.get("end_time")
        }

    # ---------------------------------------
    # Feature Processing (GLOBAL FEATURES)
    # ---------------------------------------
    def _process_features(self, gesture: Dict) -> np.ndarray:
        """معالجة الميزات العالمية (2D) مثل التدريب"""
        # Extract GLOBAL features (not temporal sequence)
        features = self.feature_extractor.gesture_to_feature_vector(gesture)

        if features is None or features.size == 0:
            raise ValueError("❌ Failed to extract features from gesture")

        # Check feature dimension
        if len(features.shape) != 1:
            raise ValueError(f"❌ Expected 1D features, got shape: {features.shape}")

        # Apply scaling (same as training)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        return features_scaled

    # ---------------------------------------
    # Prediction with Confidence Scoring
    # ---------------------------------------
    def _calculate_confidence_score(self, probabilities: np.ndarray, top_prediction: float) -> float:
        """حساب درجة الثقة بناءً على توزيع الاحتمالات"""
        confidence = top_prediction
        
        # Apply penalty if second-best is too close
        sorted_probs = np.sort(probabilities)[::-1]
        if len(sorted_probs) > 1:
            margin = sorted_probs[0] - sorted_probs[1]
            if margin < 0.3:
                confidence *= 0.7
        
        # Apply penalty for low maximum probability
        if top_prediction < 0.6:
            confidence *= 0.8
        
        return min(confidence, 1.0)

    # ---------------------------------------
    # MAIN PREDICTION
    # ---------------------------------------
    def predict_gesture(self, gesture_from_frontend: Dict[str, Any], return_topk: int = 3) -> Dict[str, Any]:
        """
        التنبؤ بالإيماءة باستخدام الميزات العالمية
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

            # 3) Process GLOBAL features
            features = self._process_features(gesture_ready)

            if self.verbose:
                print(f"📐 Input shape to model: {features.shape}")

            # 4) Predict
            preds = self.model.predict(features, verbose=0)[0]

            # 5) Process results
            pred_idx = int(np.argmax(preds))
            top_probability = float(preds[pred_idx])
            
            # Calculate enhanced confidence
            confidence = self._calculate_confidence_score(preds, top_probability)

            # 6) Decode label
            try:
                predicted_char = self.label_encoder.inverse_transform([pred_idx])[0]
            except Exception as e:
                predicted_char = f"Class_{pred_idx}"
                if self.verbose:
                    print(f"⚠️ Label decoding warning: {e}")

            # 7) Top-k predictions
            top_k = min(return_topk, len(preds))
            top_indices = preds.argsort()[::-1][:top_k]

            top_predictions = []
            for i in top_indices:
                try:
                    label = self.label_encoder.inverse_transform([int(i)])[0]
                except:
                    label = f"Class_{i}"
                
                top_predictions.append({
                    "letter": label,
                    "probability": float(preds[i])
                })

            # 8) Return comprehensive result
            result = {
                "success": True,
                "predicted_letter": predicted_char,
                "confidence": round(confidence, 3),
                "raw_confidence": round(top_probability, 3),
                "top_predictions": top_predictions,
                "quality_report": {
                    "is_valid": quality_report["is_valid"],
                    "warnings": quality_report["warnings"],
                    "frame_count": len(gesture_ready.get("frames", [])),
                    "total_points": sum(len(f.get("points", [])) for f in gesture_ready.get("frames", []))
                }
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
            "num_classes": len(self.label_encoder.classes_),
            "classes": self.label_encoder.classes_.tolist(),
            "feature_names": self.feature_extractor.get_feature_names()
        }