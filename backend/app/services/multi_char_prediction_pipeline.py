# app/services/multi_char_prediction_pipeline.py (IMPROVED)
import numpy as np
from typing import List, Dict, Any
from app.services.prediction_pipeline import PredictionPipeline
from app.services.gesture_segmenter import GestureSegmenter

class MultiCharPredictionPipeline:
    def __init__(self, model_path: str = "ai_model/final_model.keras", verbose: bool = True):
        self.single_char_predictor = PredictionPipeline(model_path=model_path, verbose=verbose)
        self.segmenter = GestureSegmenter(
            min_segment_frames=8,
            velocity_threshold=0.03,
            pause_threshold=0.015,
            direction_change_threshold=0.6
        )
        self.verbose = verbose
    
    def predict_gesture(self, gesture_from_frontend: Dict[str, Any]) -> Dict[str, Any]:
        """التنبؤ مع تحسين كشف الأحرف المتعددة"""
        
        gesture_ready = self.single_char_predictor._convert_frontend_to_training_format(gesture_from_frontend)
        complexity_analysis = self.segmenter.analyze_gesture_complexity(gesture_ready["frames"])
        
        if self.verbose:
            print(f"🔍 Gesture Analysis: {complexity_analysis}")
        
        # محاولة التجزئة حتى لو قال إنها وحيدة (للإيماءات الطويلة)
        if complexity_analysis["total_frames"] > 20:
            segments = self.segmenter.detect_segments(gesture_ready["frames"])
            
            if self.verbose:
                print(f"✂️ Detected {len(segments)} segments")
            
            if len(segments) > 1:
                return self._predict_multi_char_gesture(gesture_from_frontend, segments, complexity_analysis)
        
        # إذا فشل الكشف، جرب التنبؤ كإيماءة وحيدة
        return self._predict_single_gesture(gesture_from_frontend, complexity_analysis)
    
    def _predict_single_gesture(self, gesture: Dict[str, Any], analysis: Dict) -> Dict[str, Any]:
        """معالجة إيماءة وحيدة"""
        result = self.single_char_predictor.predict_gesture(gesture)
        result["gesture_analysis"] = analysis
        result["type"] = "single_character"
        return result
    
    def _predict_multi_char_gesture(self, gesture: Dict[str, Any], segments: List[List[Dict]], analysis: Dict) -> Dict[str, Any]:
        """معالجة إيماءة متعددة الأحرف"""
        
        segment_predictions = []
        all_predictions = []
        
        for i, segment_frames in enumerate(segments):
            if len(segment_frames) < 5:  # تجاهل المقاطع القصيرة جداً
                continue
                
            segment_gesture = {
                "frames": segment_frames,
                "duration_ms": sum(f.get("delta_ms", 16) for f in segment_frames),
                "start_time": segment_frames[0].get("timestamp", 0),
                "end_time": segment_frames[-1].get("timestamp", 0)
            }
            
            # التنبؤ بالمقطع
            segment_result = self.single_char_predictor.predict_gesture(segment_gesture)
            
            if segment_result["success"] and segment_result["confidence"] > 0.3:
                segment_prediction = {
                    "segment_index": i,
                    "frames_count": len(segment_frames),
                    "duration_ms": segment_gesture["duration_ms"],
                    "prediction": segment_result
                }
                
                segment_predictions.append(segment_prediction)
                all_predictions.append({
                    "letter": segment_result["predicted_letter"],
                    "confidence": segment_result["confidence"],
                    "segment": i
                })
        
        # إذا لم نحصل على تنبؤات كافية، نعود للتنبؤ الوحيد
        if len(all_predictions) < 2:
            if self.verbose:
                print("⚠️ Not enough valid segments, falling back to single prediction")
            return self._predict_single_gesture(gesture, analysis)
        
        # ترتيب التوقعات وبناء النتيجة
        all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        combined_text = "".join([p["letter"] for p in all_predictions[:2]])
        
        return {
            "success": True,
            "type": "multi_character",
            "combined_prediction": combined_text,
            "individual_letters": [p["letter"] for p in all_predictions[:2]],
            "segment_predictions": segment_predictions,
            "top_individual_predictions": all_predictions[:3],
            "gesture_analysis": analysis,
            "segment_count": len(segments),
            "confidence": np.mean([p["confidence"] for p in all_predictions[:2]]) if all_predictions else 0.0
        }