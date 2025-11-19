# app/services/gesture_segmenter.py (FIXED VERSION)
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

class GestureSegmenter:
    def __init__(self, 
                 min_segment_frames: int = 8,
                 velocity_threshold: float = 0.05,
                 pause_threshold: float = 0.02,
                 direction_change_threshold: float = 0.7):
        self.min_segment_frames = min_segment_frames
        self.velocity_threshold = velocity_threshold
        self.pause_threshold = pause_threshold
        self.direction_change_threshold = direction_change_threshold
    
    def _convert_to_python_types(self, obj: Any) -> Any:
        """تحويل بيانات NumPy لـ Python types عادية"""
        if isinstance(obj, (np.bool_, bool)):  # إزالة np.bool8
            return bool(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16, float)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return [self._convert_to_python_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_to_python_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_python_types(item) for item in obj]
        else:
            return obj
    
    def analyze_gesture_complexity(self, frames: List[Dict]) -> Dict:
        """تحليل تعقيد الإيماءة مع تحويل الأنواع"""
        if not frames:
            return self._convert_to_python_types({
                "is_multi_char": False, 
                "segment_count": 1, 
                "confidence": 0.0
            })
        
        # إحصائيات متقدمة
        velocities = []
        accelerations = []
        direction_changes = []
        pauses = 0
        
        for i in range(1, len(frames)):
            # السرعة
            vel = self._calculate_frame_velocity(frames[i-1], frames[i])
            velocities.append(vel)
            
            # التسارع
            if i > 1:
                prev_vel = self._calculate_frame_velocity(frames[i-2], frames[i-1])
                accel = abs(vel - prev_vel) / max(prev_vel, 0.001)
                accelerations.append(accel)
            
            # تغير الاتجاه
            if i > 2:
                dir_change = self._calculate_direction_change(frames[i-3], frames[i-2], frames[i-1])
                direction_changes.append(dir_change)
            
            # التوقف
            if vel < self.pause_threshold:
                pauses += 1
        
        # تحويل القيم لـ Python floats فوراً
        avg_velocity = float(np.mean(velocities)) if velocities else 0.0
        velocity_std = float(np.std(velocities)) if velocities else 0.0
        pause_ratio = float(pauses / len(frames)) if frames else 0.0
        significant_direction_changes = len([d for d in direction_changes if d > 0.3])
        
        # معايير متعددة الأحرف
        length_criterion = len(frames) > 25
        velocity_criterion = velocity_std > avg_velocity * 0.5
        pause_criterion = pause_ratio > 0.1
        direction_criterion = significant_direction_changes > len(frames) * 0.08
        
        is_multi_char = (
            length_criterion and 
            (velocity_criterion or pause_criterion or direction_criterion)
        )
        
        # حساب الثقة
        confidence_factors = []
        if length_criterion: confidence_factors.append(0.3)
        if velocity_criterion: confidence_factors.append(0.3)
        if pause_criterion: confidence_factors.append(0.2)
        if direction_criterion: confidence_factors.append(0.2)
        
        confidence = float(sum(confidence_factors)) if confidence_factors else 0.0
        
        result = {
            "is_multi_char": bool(is_multi_char),  # تحويل لـ bool مباشرة
            "segment_count": 2 if is_multi_char else 1,
            "confidence": min(confidence, 1.0),
            "total_frames": int(len(frames)),  # تحويل لـ int
            "avg_velocity": avg_velocity,
            "velocity_variation": velocity_std,
            "pause_ratio": pause_ratio,
            "direction_changes": int(significant_direction_changes),  # تحويل لـ int
            "criteria_met": {
                "length": bool(length_criterion),
                "velocity_variation": bool(velocity_criterion),
                "pauses": bool(pause_criterion),
                "direction_changes": bool(direction_criterion)
            }
        }
        
        return result
    

    def detect_segments(self, frames: List[Dict]) -> List[List[Dict]]:
        """كشف المقاطع في الإيماءة"""
        if len(frames) < self.min_segment_frames * 2:
            return [frames]
        
        segments = []
        current_segment = [frames[0]]
        in_pause = False
        pause_start = 0
        
        for i in range(1, len(frames)):
            current_frame = frames[i]
            prev_frame = frames[i-1]
            
            # حساب السرعة والتوقف
            velocity = self._calculate_frame_velocity(prev_frame, current_frame)
            is_pause = velocity < self.pause_threshold
            
            # اكتشاف التوقف الطويل
            if is_pause:
                if not in_pause:
                    in_pause = True
                    pause_start = i
                else:
                    pause_duration = i - pause_start
                    if pause_duration >= 3 and len(current_segment) >= self.min_segment_frames:
                        segments.append(current_segment)
                        current_segment = [current_frame]
                        in_pause = False
            else:
                in_pause = False
                
                if velocity < self.velocity_threshold and len(current_segment) >= self.min_segment_frames:
                    if i > 2:
                        direction_change = self._calculate_direction_change(frames[i-3], frames[i-2], frames[i-1])
                        if direction_change > self.direction_change_threshold:
                            segments.append(current_segment)
                            current_segment = [current_frame]
                            continue
                
                current_segment.append(current_frame)
        
        if current_segment:
            segments.append(current_segment)
        
        if len(segments) == 1 and len(frames) > 20:
            forced_segments = self._force_segmentation(frames)
            if len(forced_segments) > 1:
                return forced_segments
        
        return segments if len(segments) > 1 else [frames]
    
    def _force_segmentation(self, frames: List[Dict]) -> List[List[Dict]]:
        """تقسيم إجباري للإيماءات الطويلة"""
        if len(frames) < 15:
            return [frames]
        
        mid_point = len(frames) // 2
        best_split = mid_point
        min_velocity = float('inf')
        
        for i in range(max(5, mid_point - 5), min(len(frames) - 5, mid_point + 5)):
            velocity = self._calculate_frame_velocity(frames[i-1], frames[i])
            if velocity < min_velocity:
                min_velocity = velocity
                best_split = i
        
        return [frames[:best_split], frames[best_split:]]
    
    def _calculate_frame_velocity(self, frame1: Dict, frame2: Dict) -> float:
        """حساب السرعة بين إطارين"""
        points1 = frame1.get("points", [])
        points2 = frame2.get("points", [])
        
        if not points1 or not points2:
            return 0.0
        
        total_movement = 0.0
        point_pairs = min(len(points1), len(points2))
        
        for j in range(point_pairs):
            x1, y1 = points1[j].get("x", 0.0), points1[j].get("y", 0.0)
            x2, y2 = points2[j].get("x", 0.0), points2[j].get("y", 0.0)
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_movement += distance
        
        avg_movement = total_movement / point_pairs if point_pairs > 0 else 0.0
        time_diff = max(frame2.get("delta_ms", 16) / 1000.0, 0.001)
        
        return float(avg_movement / time_diff)
    
    def _calculate_direction_change(self, frame1: Dict, frame2: Dict, frame3: Dict) -> float:
        """حساب تغير الاتجاه"""
        points1, points2, points3 = (
            frame1.get("points", []),
            frame2.get("points", []),
            frame3.get("points", [])
        )
        
        if not all([points1, points2, points3]):
            return 0.0
        
        centroid1 = self._calculate_centroid(points1)
        centroid2 = self._calculate_centroid(points2)  
        centroid3 = self._calculate_centroid(points3)
        
        vec1 = (centroid2[0] - centroid1[0], centroid2[1] - centroid1[1])
        vec2 = (centroid3[0] - centroid2[0], centroid3[1] - centroid2[1])
        
        dot_product = vec1[0]*vec2[0] + vec1[1]*vec2[1]
        mag1 = np.sqrt(vec1[0]**2 + vec1[1]**2)
        mag2 = np.sqrt(vec2[0]**2 + vec2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return float(angle / np.pi)
    
    def _calculate_centroid(self, points: List[Dict]) -> Tuple[float, float]:
        """حساب المركز الهندسي"""
        if not points:
            return (0.0, 0.0)
        
        xs = [p.get("x", 0.0) for p in points]
        ys = [p.get("y", 0.0) for p in points]
        
        return (float(np.mean(xs)), float(np.mean(ys)))
        """حساب المركز الهندسي"""
        if not points:
            return (0.0, 0.0)
        
        xs = [p.get("x", 0.0) for p in points]
        ys = [p.get("y", 0.0) for p in points]
        
        return (float(np.mean(xs)), float(np.mean(ys)))