# app/services/advanced_feature_extractor.py
from typing import List, Dict, Any, Tuple
import numpy as np
import csv
import matplotlib.pyplot as plt

class AdvancedFeatureExtractor:
    def __init__(self, max_timesteps: int = 200, verbose: bool = False):
        self.max_timesteps = max_timesteps
        self.verbose = verbose
        
        # الميزات المحسنة
        self.per_frame_names = [
            # الميزات الحركية عالية التباين
            "velocity_magnitude", "acceleration_magnitude",
            "angular_velocity", "curvature",
            
            # الميزات المكانية المفيدة
            "centroid_x", "centroid_y",
            "spread_x", "spread_y",
            
            # الميزات الزمنية
            "progress", "time_from_start",
            
            # الميزات الشكلية المحسنة
            "compactness", "aspect_ratio"
        ]

    def _compute_curvature(self, points: List[Dict]) -> float:
        """حساب انحناء الرسمة"""
        if len(points) < 3:
            return 0.0
        
        xs = np.array([p["x"] for p in points])
        ys = np.array([p["y"] for p in points])
        
        dx_dt = np.gradient(xs)
        dy_dt = np.gradient(ys)
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        
        numerator = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
        denominator = np.power(dx_dt**2 + dy_dt**2, 1.5)
        
        curvature = np.mean(numerator / (denominator + 1e-8))
        return float(curvature)

    def _compute_compactness(self, points: List[Dict]) -> float:
        """حساب كثافة الرسمة"""
        if len(points) < 3:
            return 0.0
        
        xs = np.array([p["x"] for p in points])
        ys = np.array([p["y"] for p in points])
        
        area = 0.5 * np.abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
        
        dx = np.diff(xs, append=xs[0])
        dy = np.diff(ys, append=ys[0])
        perimeter = np.sum(np.sqrt(dx**2 + dy**2))
        
        if perimeter == 0:
            return 0.0
            
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        return float(compactness)

    def _compute_aspect_ratio(self, points: List[Dict]) -> float:
        """حساب نسبة الطول إلى العرض"""
        if not points:
            return 1.0
            
        xs = np.array([p["x"] for p in points])
        ys = np.array([p["y"] for p in points])
        
        width = np.max(xs) - np.min(xs)
        height = np.max(ys) - np.min(ys)
        
        if height == 0:
            return float('inf') if width > 0 else 1.0
            
        return float(width / height)

    def _extract_optimized_features(self, frame: Dict, prev_frame: Dict, 
                                  prev_velocity: np.ndarray, 
                                  cumulative_length: float,
                                  start_time: float,
                                  global_centroid: np.ndarray) -> Dict:
        """استخراج الميزات عالية التباين فقط"""
        pts = frame.get("points", [])
        timestamp = frame.get("timestamp", 0)
        delta_s = max(frame.get("delta_ms", 1) / 1000.0, 1e-6)

        if not pts:
            return {name: 0.0 for name in self.per_frame_names}

        # استخراج الإحداثيات
        x_vals = np.array([p["x"] for p in pts])
        y_vals = np.array([p["y"] for p in pts])
        
        # الميزات المكانية الأساسية
        centroid_x, centroid_y = float(np.mean(x_vals)), float(np.mean(y_vals))
        spread_x, spread_y = float(np.std(x_vals)), float(np.std(y_vals))
        
        # الميزات الحركية (عالية التباين)
        velocity_magnitude = 0.0
        acceleration_magnitude = 0.0
        angular_velocity = 0.0
        
        if prev_frame is not None:
            prev_pts = prev_frame.get("points", [])
            if prev_pts:
                prev_x = np.mean([p["x"] for p in prev_pts])
                prev_y = np.mean([p["y"] for p in prev_pts])
                
                velocity_x = (centroid_x - prev_x) / delta_s
                velocity_y = (centroid_y - prev_y) / delta_s
                velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
                
                # التسارع
                if prev_velocity is not None:
                    acceleration_x = (velocity_x - prev_velocity[0]) / delta_s
                    acceleration_y = (velocity_y - prev_velocity[1]) / delta_s
                    acceleration_magnitude = np.sqrt(acceleration_x**2 + acceleration_y**2)
                
                # السرعة الزاوية
                if len(pts) > 1 and len(prev_pts) > 1:
                    current_angle = np.arctan2(pts[-1]["y"] - pts[0]["y"], 
                                             pts[-1]["x"] - pts[0]["x"])
                    prev_angle = np.arctan2(prev_pts[-1]["y"] - prev_pts[0]["y"], 
                                          prev_pts[-1]["x"] - prev_pts[0]["x"])
                    angular_velocity = (current_angle - prev_angle) / delta_s

        # الميزات الشكلية
        curvature = self._compute_curvature(pts)
        compactness = self._compute_compactness(pts)
        aspect_ratio = self._compute_aspect_ratio(pts)

        # الميزات الزمنية
        time_from_start = (timestamp - start_time) / 1000.0
        
        # المسافة التراكمية والتقدم
        if prev_frame is not None:
            prev_centroid_x = np.mean([p["x"] for p in prev_frame.get("points", [])])
            prev_centroid_y = np.mean([p["y"] for p in prev_frame.get("points", [])])
            segment_length = np.sqrt((centroid_x - prev_centroid_x)**2 + 
                                   (centroid_y - prev_centroid_y)**2)
            cumulative_length += segment_length

        progress = cumulative_length

        return {
            # حركية عالية التباين
            "velocity_magnitude": velocity_magnitude,
            "acceleration_magnitude": acceleration_magnitude,
            "angular_velocity": angular_velocity,
            "curvature": curvature,
            
            # مكانية
            "centroid_x": centroid_x, "centroid_y": centroid_y,
            "spread_x": spread_x, "spread_y": spread_y,
            
            # زمنية
            "progress": progress, 
            "time_from_start": time_from_start,
            
            # شكلية محسنة
            "compactness": compactness,
            "aspect_ratio": aspect_ratio
        }

    def _gesture_to_sequence(self, gesture: Dict) -> np.ndarray:
        """تحويل الإيماءة إلى تسلسل الميزات المحسن"""
        frames = gesture.get("frames", [])
        if not frames:
            return np.zeros((self.max_timesteps, len(self.per_frame_names)), dtype=np.float32)

        # حساب المركز العالمي
        all_x = [p["x"] for f in frames for p in f.get("points", [])]
        all_y = [p["y"] for f in frames for p in f.get("points", [])]
        global_centroid = np.array([
            float(np.mean(all_x)) if all_x else 0.0,
            float(np.mean(all_y)) if all_y else 0.0
        ])

        start_time = frames[0].get("timestamp", 0) if frames else 0

        # استخراج الميزات
        feature_buffer = []
        prev_frame = None
        prev_velocity = None
        cumulative_length = 0.0

        for i, frame in enumerate(frames):
            features = self._extract_optimized_features(
                frame=frame,
                prev_frame=prev_frame,
                prev_velocity=prev_velocity,
                cumulative_length=cumulative_length,
                start_time=start_time,
                global_centroid=global_centroid
            )
            
            if prev_frame is not None:
                cumulative_length = features["progress"]
            
            prev_velocity = np.array([0.0, 0.0])
            prev_frame = frame
            feature_buffer.append(features)

        # تطبيع التقدم
        total_length = cumulative_length if cumulative_length > 0 else 1.0
        for features in feature_buffer:
            features["progress"] = features["progress"] / total_length

        # تحويل إلى مصفوفة
        sequence_array = np.array([[features[name] for name in self.per_frame_names] 
                                 for features in feature_buffer], dtype=np.float32)

        T, D = sequence_array.shape

        # إعادة العينات أو الحشو
        if T >= self.max_timesteps:
            indices = np.linspace(0, T - 1, self.max_timesteps).astype(int)
            sequence_array = sequence_array[indices]
        else:
            padding = np.zeros((self.max_timesteps - T, D), dtype=np.float32)
            sequence_array = np.vstack([sequence_array, padding])

        return sequence_array

    # 🔥 هذه هي الدالة التي يحتاجها TrainingPipeline
    def gesture_to_full_feature_vector(self, gesture: Dict) -> np.ndarray:
        """
        واجهة متوافقة مع TrainingPipeline
        تحويل الإيماءة إلى متجه ميزات كامل
        """
        return self._gesture_to_sequence(gesture)

    def save_gestures_to_csv(self, gestures: List[Dict[str, Any]], 
                           out_csv: str = "ai_model/ADVANCED_features.csv"):
        """حفظ الميزات إلى ملف CSV"""
        if not gestures:
            print("❌ No gestures to process.")
            return
            
        print(f"📊 Processing {len(gestures)} gestures for feature extraction...")
        
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["gesture_id", "character"] + self.per_frame_names
            writer.writerow(header)
            
            successful = 0
            for i, gesture in enumerate(gestures):
                try:
                    sequence = self.gesture_to_full_feature_vector(gesture)
                    mean_features = np.mean(sequence, axis=0).tolist()
                    
                    writer.writerow([
                        gesture.get("id", f"unknown_{i}"),
                        gesture.get("character", "unknown")
                    ] + mean_features)
                    
                    successful += 1
                    
                    if self.verbose and (i + 1) % 50 == 0:
                        print(f"✅ Processed {i + 1}/{len(gestures)} gestures")
                        
                except Exception as e:
                    print(f"⚠️ Error processing gesture {gesture.get('id')}: {e}")
                    continue
        
        print(f"🎉 Saved features for {successful}/{len(gestures)} gestures to {out_csv}")

    def get_feature_names(self) -> List[str]:
        """الحصول على أسماء الميزات"""
        return self.per_frame_names

    def get_feature_dimension(self) -> int:
        """الحصول على بعد الميزات"""
        return len(self.per_frame_names)