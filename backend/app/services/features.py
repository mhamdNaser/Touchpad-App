import numpy as np
from typing import List, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def extract_features(self, gestures_data: List[Dict]) -> tuple:
        """
        استخراج الميزات من بيانات الإيماءات
        """
        features = []
        labels = []
        
        for gesture in gestures_data:
            gesture_features = self.extract_gesture_features(gesture)
            if gesture_features is not None:
                features.append(gesture_features)
                labels.append(gesture['character'])
        
        return np.array(features), np.array(labels)
    
    def extract_gesture_features(self, gesture: Dict) -> np.ndarray:
        """
        استخراج ميزات إيماءة واحدة
        """
        if not gesture['frames']:
            return None
        
        # 1. ميزات المسار (Trajectory Features)
        trajectory_features = self.extract_trajectory_features(gesture)
        
        # 2. ميزات السرعة والتسارع
        velocity_features = self.extract_velocity_features(gesture)
        
        # 3. ميزات الشكل (Shape Features)
        shape_features = self.extract_shape_features(gesture)
        
        # 4. ميزات الوقت
        time_features = self.extract_time_features(gesture)
        
        # دمج جميع الميزات
        all_features = np.concatenate([
            trajectory_features,
            velocity_features,
            shape_features,
            time_features
        ])
        
        return all_features
    
    def extract_trajectory_features(self, gesture: Dict) -> np.ndarray:
        """استخراج ميزات المسار"""
        points = self.get_all_points(gesture)
        if len(points) < 2:
            return np.zeros(10)
        
        x_coords = [p['x'] for p in points]
        y_coords = [p['y'] for p in points]
        
        features = [
            np.mean(x_coords), np.std(x_coords),  # متوسط وانحراف إحداثيات X
            np.mean(y_coords), np.std(y_coords),  # متوسط وانحراف إحداثيات Y
            np.max(x_coords) - np.min(x_coords),  # عرض الإيماءة
            np.max(y_coords) - np.min(y_coords),  # ارتفاع الإيماءة
            np.mean(np.sqrt(np.array(x_coords)**2 + np.array(y_coords)**2)),  # متوسط المسافة من الأصل
        ]
        
        return np.array(features)
    
    def extract_velocity_features(self, gesture: Dict) -> np.ndarray:
        """استخراج ميزات السرعة"""
        # سيتم تطبيق هذا في الخطوات القادمة
        return np.zeros(5)
    
    def extract_shape_features(self, gesture: Dict) -> np.ndarray:
        """استخراج ميزات الشكل"""
        return np.zeros(3)
    
    def extract_time_features(self, gesture: Dict) -> np.ndarray:
        """استخراج ميزات الوقت"""
        return np.array([
            gesture['duration_ms'],
            gesture['frame_count'],
            gesture['duration_ms'] / max(gesture['frame_count'], 1)
        ])
    
    def get_all_points(self, gesture: Dict) -> List[Dict]:
        """جمع جميع النقاط من جميع الإطارات"""
        all_points = []
        for frame in gesture['frames']:
            all_points.extend(frame['points'])
        return all_points