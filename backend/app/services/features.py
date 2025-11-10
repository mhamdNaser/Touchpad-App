import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict

class FeatureEngineer:
    def __init__(self, max_timesteps: int = 100):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.max_timesteps = max_timesteps

    ###############################################
    ##        الخطوة 1: استخراج الفيتشر          ##
    ###############################################
    def extract_features(self, gestures_data: List[Dict]) -> tuple:
        features = []
        labels = []

        for gesture in gestures_data:
            sequence = self.extract_sequence_features(gesture)
            if sequence is not None:
                features.append(sequence)
                labels.append(gesture['character'])

        X = np.array(features)
        y = np.array(labels)

        if len(X.shape) < 3:
            print("⚠️ لا توجد بيانات كافية لاستخراج الميزات.")
            return None, None

        # تطبيع القيم
        n_samples, timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(n_samples, timesteps, n_features)

        # ترميز الأحرف
        y_encoded = self.label_encoder.fit_transform(y)

        return X, y_encoded

    ###############################################
    ##    الخطوة 2: استخراج الميزات لكل فريم      ##
    ###############################################
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
            sequence.extend([[0] * feature_dim] * pad_len)
        else:
            sequence = sequence[:self.max_timesteps]

        return np.array(sequence)

    ###############################################
    ##   الخطوة 3: تجميع الفيتشر لكل حرف         ##
    ###############################################
    def aggregate_by_character(self, gestures_data: List[Dict]) -> Dict[str, np.ndarray]:
        char_features = defaultdict(list)

        for gesture in gestures_data:
            seq = self.extract_sequence_features(gesture)
            if seq is not None:
                flat = np.mean(seq, axis=0)
                char_features[gesture['character']].append(flat)

        aggregated = {}
        for char, feats in char_features.items():
            aggregated[char] = np.mean(np.stack(feats), axis=0)

        return aggregated

    ###############################################
    ##   الخطوة 4: طباعة الفيتشر في شكل جدول     ##
    ###############################################
    def show_feature_table(self, aggregated_features: Dict[str, np.ndarray]):
        """
        طباعة الفيتشر المجمعة لكل حرف في شكل جدول باستخدام pandas
        """
        feature_names = [
            "mean_x", "std_x",
            "mean_y", "std_y",
            "mean_pressure", "std_pressure",
            "points_count"
        ]

        df = pd.DataFrame.from_dict(aggregated_features, orient='index', columns=feature_names)
        df.index.name = "Character"

        print("\n================= 📊 FEATURE TABLE =================\n")
        print(df.round(3))
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
