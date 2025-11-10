# app/services/features.py
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FeatureEngineer:
    def __init__(self, max_timesteps: int = 100):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.max_timesteps = max_timesteps

    # ===== دوال LSTM =====
    def extract_sequence_features(self, gesture: Dict) -> np.ndarray:
        frames = gesture.get('frames', [])
        if not frames:
            return None

        sequence = []
        for frame in frames:
            points = frame.get('points', [])
            if not points:
                continue

            x_coords = [p.get('x', 0.0) for p in points]
            y_coords = [p.get('y', 0.0) for p in points]
            pressure = [p.get('pressure', 0.0) for p in points]

            frame_features = [
                np.mean(x_coords), np.std(x_coords),
                np.mean(y_coords), np.std(y_coords),
                np.mean(pressure), np.std(pressure),
                len(points)
            ]
            sequence.append(frame_features)

        if not sequence:
            return None

        feature_dim = len(sequence[0])
        if len(sequence) < self.max_timesteps:
            pad_len = self.max_timesteps - len(sequence)
            sequence.extend([[0]*feature_dim]*pad_len)
        else:
            sequence = sequence[:self.max_timesteps]

        return np.array(sequence)

    def extract_features(self, gestures_data: List[Dict]):
        features = []
        labels = []

        for gesture in gestures_data:
            seq = self.extract_sequence_features(gesture)
            if seq is not None:
                features.append(seq)
                labels.append(gesture['character'])

        X = np.array(features)
        y = np.array(labels)

        # تطبيع الميزات
        n_samples, timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(n_samples, timesteps, n_features)

        # ترميز الأحرف
        y_encoded = self.label_encoder.fit_transform(y)

        return X, y_encoded

    # ===== دوال تحليل الفيتشر لكل حرف =====
    def aggregate_by_character(self, gestures_data: List[Dict]) -> Dict:
        """
        تجميع الفيتشر لكل حرف لحساب المتوسط والانحراف المعياري
        """
        agg = {}
        for gesture in gestures_data:
            char = gesture['character']
            if char not in agg:
                agg[char] = {
                    'mean_x': [], 'std_x': [],
                    'mean_y': [], 'std_y': [],
                    'mean_pressure': [], 'std_pressure': [],
                    'points_count': []
                }

            for frame in gesture.get('frames', []):
                points = frame.get('points', [])
                if not points:
                    continue
                x_coords = [p.get('x', 0.0) for p in points]
                y_coords = [p.get('y', 0.0) for p in points]
                pressure = [p.get('pressure', 0.0) for p in points]

                agg[char]['mean_x'].append(np.mean(x_coords))
                agg[char]['std_x'].append(np.std(x_coords))
                agg[char]['mean_y'].append(np.mean(y_coords))
                agg[char]['std_y'].append(np.std(y_coords))
                agg[char]['mean_pressure'].append(np.mean(pressure))
                agg[char]['std_pressure'].append(np.std(pressure))
                agg[char]['points_count'].append(len(points))

        return agg

    def show_feature_table(self, aggregated_features: Dict) -> pd.DataFrame:
        """
        إنشاء جدول تحليلي للفيتشر لكل حرف
        """
        rows = []
        for char, feats in aggregated_features.items():
            rows.append({
                'Character': char,
                'mean_x': np.mean(feats['mean_x']),
                'std_x': np.mean(feats['std_x']),
                'mean_y': np.mean(feats['mean_y']),
                'std_y': np.mean(feats['std_y']),
                'mean_pressure': np.mean(feats['mean_pressure']),
                'std_pressure': np.mean(feats['std_pressure']),
                'points_count': np.mean(feats['points_count'])
            })
        df = pd.DataFrame(rows).set_index('Character')
        print("\n================= 📊 FEATURE TABLE =================\n")
        print(df)
        print("\n====================================================\n")
        return df





# import numpy as np
# from typing import List, Dict
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# class FeatureEngineer:
#     def __init__(self, max_timesteps: int = 100):
#         self.scaler = StandardScaler()
#         self.label_encoder = LabelEncoder()
#         self.max_timesteps = max_timesteps  # أقصى عدد إطارات لكل إيماءة (لتوحيد الطول)
    
#     def extract_features(self, gestures_data: List[Dict]) -> tuple:
#         """
#         تحويل بيانات الإيماءات إلى تسلسل عددي مناسب لـ LSTM
#         """
#         features = []
#         labels = []
        
#         for gesture in gestures_data:
#             sequence = self.extract_sequence_features(gesture)
#             if sequence is not None:
#                 features.append(sequence)
#                 labels.append(gesture['character'])
        
#         X = np.array(features)
#         y = np.array(labels)
        
#         # تطبيع الميزات
#         n_samples, timesteps, n_features = X.shape
#         X_reshaped = X.reshape(-1, n_features)
#         X_scaled = self.scaler.fit_transform(X_reshaped)
#         X = X_scaled.reshape(n_samples, timesteps, n_features)

#         # ترميز الأحرف إلى أرقام
#         y_encoded = self.label_encoder.fit_transform(y)

#         return X, y_encoded
    
#     def extract_sequence_features(self, gesture: Dict) -> np.ndarray:
#         """
#         استخراج ميزات لكل إطار (frame) ضمن الإيماءة
#         """
#         frames = gesture['frames']
#         if not frames:
#             return None
        
#         sequence = []
#         for frame in frames:
#             if not frame['points']:
#                 continue

#             x_coords = [p['x'] for p in frame['points']]
#             y_coords = [p['y'] for p in frame['points']]
#             pressure = [p['pressure'] for p in frame['points']]
            
#             # ميزات لكل إطار (frame-level features)
#             frame_features = [
#                 np.mean(x_coords), np.std(x_coords),
#                 np.mean(y_coords), np.std(y_coords),
#                 np.mean(pressure), np.std(pressure),
#                 len(frame['points'])
#             ]
#             sequence.append(frame_features)
        
#         # توحيد الطول (padding/truncation)
#         if len(sequence) < self.max_timesteps:
#             pad_len = self.max_timesteps - len(sequence)
#             sequence.extend([[0]*len(sequence[0])] * pad_len)
#         else:
#             sequence = sequence[:self.max_timesteps]
        
#         return np.array(sequence)
