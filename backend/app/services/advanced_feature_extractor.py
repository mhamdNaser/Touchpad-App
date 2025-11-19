# app/services/advanced_feature_extractor.py (IMPROVED VERSION)
from typing import List, Dict, Any, Tuple
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import stats

class AdvancedFeatureExtractor:
    def __init__(self, max_timesteps: int = 60, verbose: bool = True):  # تقليل الإطارات
        self.max_timesteps = max_timesteps
        self.verbose = verbose
        
        # ميزات أكثر تمييزاً للحروف العربية
        self.feature_names = [
            # الميزات الأساسية
            "centroid_x", "centroid_y",
            "bounding_width", "bounding_height",
            "aspect_ratio",
            
            # الميزات الحركية
            "total_velocity", "max_velocity", "velocity_std",
            "total_acceleration", "max_acceleration", 
            "curvature", "angular_velocity",
            
            # ميزات التوقيت
            "duration", "drawing_speed",
            
            # ميزات الشكل
            "compactness", "complexity",
            "start_end_distance", "centroid_distance"
        ]

    def _extract_global_features(self, frames: List[Dict]) -> Dict[str, float]:
        """استخراج الميزات العالمية للإيماءة"""
        if not frames:
            return {name: 0.0 for name in self.feature_names}
        
        all_points = []
        all_velocities = []
        all_accelerations = []
        timestamps = []
        
        # جمع جميع النقاط والبيانات الزمنية
        for frame in frames:
            points = frame.get("points", [])
            if points:
                # المركز الحالي
                centroid_x = np.mean([p["x"] for p in points])
                centroid_y = np.mean([p["y"] for p in points])
                all_points.append([centroid_x, centroid_y])
                
                # السرعة
                frame_velocity = np.mean([p.get("speed", 0.0) for p in points])
                all_velocities.append(frame_velocity)
                
                # التسارع
                frame_accel = np.mean([
                    np.sqrt(p.get("acceleration_x", 0.0)**2 + p.get("acceleration_y", 0.0)**2) 
                    for p in points
                ])
                all_accelerations.append(frame_accel)
                
            timestamps.append(frame.get("timestamp", 0))
        
        if not all_points:
            return {name: 0.0 for name in self.feature_names}
            
        points_array = np.array(all_points)
        velocities_array = np.array(all_velocities)
        accelerations_array = np.array(all_accelerations)
        
        # الميزات الأساسية
        min_x, min_y = np.min(points_array, axis=0)
        max_x, max_y = np.max(points_array, axis=0)
        width = max_x - min_x
        height = max_y - min_y
        aspect_ratio = width / height if height > 0 else 1.0
        
        # الميزات الحركية
        total_velocity = np.sum(velocities_array)
        max_velocity = np.max(velocities_array) if len(velocities_array) > 0 else 0.0
        velocity_std = np.std(velocities_array) if len(velocities_array) > 0 else 0.0
        
        total_acceleration = np.sum(accelerations_array)
        max_acceleration = np.max(accelerations_array) if len(accelerations_array) > 0 else 0.0
        
        # الانحناء
        curvature = self._compute_trajectory_curvature(points_array)
        
        # السرعة الزاوية
        angular_velocity = self._compute_angular_velocity(points_array)
        
        # ميزات التوقيت
        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 1.0
        drawing_speed = len(points_array) / duration if duration > 0 else 0.0
        
        # ميزات الشكل
        compactness = self._compute_compactness(points_array)
        complexity = self._compute_complexity(points_array)
        
        # المسافات
        start_point = points_array[0]
        end_point = points_array[-1]
        start_end_distance = np.linalg.norm(end_point - start_point)
        
        centroid = np.mean(points_array, axis=0)
        centroid_distance = np.mean([np.linalg.norm(p - centroid) for p in points_array])
        
        return {
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "bounding_width": float(width),
            "bounding_height": float(height),
            "aspect_ratio": float(aspect_ratio),
            
            "total_velocity": float(total_velocity),
            "max_velocity": float(max_velocity),
            "velocity_std": float(velocity_std),
            "total_acceleration": float(total_acceleration),
            "max_acceleration": float(max_acceleration),
            "curvature": float(curvature),
            "angular_velocity": float(angular_velocity),
            
            "duration": float(duration),
            "drawing_speed": float(drawing_speed),
            
            "compactness": float(compactness),
            "complexity": float(complexity),
            "start_end_distance": float(start_end_distance),
            "centroid_distance": float(centroid_distance)
        }

    def _compute_trajectory_curvature(self, points: np.ndarray) -> float:
        """حساب انحناء المسار"""
        if len(points) < 3:
            return 0.0
            
        curvatures = []
        for i in range(1, len(points)-1):
            p1, p2, p3 = points[i-1], points[i], points[i+1]
            
            # متجهات الاتجاه
            v1 = p2 - p1
            v2 = p3 - p2
            
            # الزاوية بين المتجهات
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 > 0 and norm_v2 > 0:
                cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                curvatures.append(angle)
        
        return float(np.mean(curvatures)) if curvatures else 0.0

    def _compute_angular_velocity(self, points: np.ndarray) -> float:
        """حساب السرعة الزاوية"""
        if len(points) < 2:
            return 0.0
            
        angles = []
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        
        if len(angles) < 2:
            return 0.0
            
        angular_changes = np.abs(np.diff(angles))
        return float(np.mean(angular_changes))

    def _compute_compactness(self, points: np.ndarray) -> float:
        """حساب الكثافة (المساحة / المحيط)^2"""
        if len(points) < 3:
            return 0.0
            
        # المساحة التقريبية
        x, y = points[:, 0], points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        # المحيط التقريبي
        dx = np.diff(x, append=x[0])
        dy = np.diff(y, append=y[0])
        perimeter = np.sum(np.sqrt(dx**2 + dy**2))
        
        if perimeter == 0:
            return 0.0
            
        return float((4 * np.pi * area) / (perimeter ** 2))

    def _compute_complexity(self, points: np.ndarray) -> float:
        """حساب تعقيد الرسمة (الانتروبيا)"""
        if len(points) < 2:
            return 0.0
            
        # حساب الزوايا بين النقاط المتتالية
        angles = []
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        
        if not angles:
            return 0.0
            
        # حساب الانتروبيا للزوايا
        hist, _ = np.histogram(angles, bins=10, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist))
        
        return float(entropy)

    def gesture_to_feature_vector(self, gesture: Dict) -> np.ndarray:
        """تحويل الإيماءة إلى متجه ميزات"""
        frames = gesture.get("frames", [])
        global_features = self._extract_global_features(frames)
        
        # تحويل إلى مصفوفة
        feature_vector = np.array([global_features[name] for name in self.feature_names], 
                                dtype=np.float32)
        
        return feature_vector

    def save_gestures_to_csv(self, gestures: List[Dict], filename: str = "features.csv"):
        """حفظ الميزات في ملف CSV"""
        if not gestures:
            print("❌ لا توجد إيماءات للمعالجة")
            return
            
        print(f"📊 جاري معالجة {len(gestures)} إيماءة...")
        
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["gesture_id", "character"] + self.feature_names
            writer.writerow(header)
            
            successful = 0
            for i, gesture in enumerate(gestures):
                try:
                    features = self.gesture_to_feature_vector(gesture)
                    
                    writer.writerow([
                        gesture.get("gesture_id", f"unknown_{i}"),
                        gesture.get("character", "unknown")
                    ] + features.tolist())
                    
                    successful += 1
                    
                    if self.verbose and (i + 1) % 50 == 0:
                        print(f"✅ تم معالجة {i + 1}/{len(gestures)}")
                        
                except Exception as e:
                    print(f"⚠️ خطأ في الإيماءة {gesture.get('gesture_id')}: {e}")
                    continue
        
        print(f"🎉 تم حفظ {successful} إيماءة في {filename}")

    def analyze_features_by_character(self, gestures: List[Dict]):
        """تحليل الميزات حسب الحرف"""
        characters = {}
        
        for gesture in gestures:
            char = gesture.get("character")
            if char not in characters:
                characters[char] = []
                
            try:
                features = self.gesture_to_feature_vector(gesture)
                characters[char].append(features)
            except Exception as e:
                continue
        
        # طباعة إحصائيات لكل حرف
        for char, feature_list in characters.items():
            if not feature_list:
                continue
                
            feature_matrix = np.array(feature_list)
            means = np.mean(feature_matrix, axis=0)
            stds = np.std(feature_matrix, axis=0)
            
            print(f"\n📊 الحرف '{char}': {len(feature_list)} عينة")
            for i, name in enumerate(self.feature_names):
                print(f"   {name}: {means[i]:.3f} ± {stds[i]:.3f}")

    def get_feature_names(self) -> List[str]:
        return self.feature_names

    def get_feature_dimension(self) -> int:
        return len(self.feature_names)