import numpy as np
from typing import List, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FeatureEngineer:
    def __init__(self, max_timesteps: int = 100):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.max_timesteps = max_timesteps  # أقصى عدد إطارات لكل إيماءة (لتوحيد الطول)
    
    def extract_features(self, gestures_data: List[Dict]) -> tuple:
        """
        تحويل بيانات الإيماءات إلى تسلسل عددي مناسب لـ LSTM
        """
        features = []
        labels = []
        
        for gesture in gestures_data:
            sequence = self.extract_sequence_features(gesture)
            if sequence is not None:
                features.append(sequence)
                labels.append(gesture['character'])
        
        X = np.array(features)
        y = np.array(labels)
        
        # تطبيع الميزات
        n_samples, timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(n_samples, timesteps, n_features)

        # ترميز الأحرف إلى أرقام
        y_encoded = self.label_encoder.fit_transform(y)

        return X, y_encoded
    
    def extract_sequence_features(self, gesture: Dict) -> np.ndarray:
        """
        استخراج ميزات لكل إطار (frame) ضمن الإيماءة
        """
        frames = gesture['frames']
        if not frames:
            return None
        
        sequence = []
        for frame in frames:
            if not frame['points']:
                continue

            x_coords = [p['x'] for p in frame['points']]
            y_coords = [p['y'] for p in frame['points']]
            pressure = [p['pressure'] for p in frame['points']]
            
            # ميزات لكل إطار (frame-level features)
            frame_features = [
                np.mean(x_coords), np.std(x_coords),
                np.mean(y_coords), np.std(y_coords),
                np.mean(pressure), np.std(pressure),
                len(frame['points'])
            ]
            sequence.append(frame_features)
        
        # توحيد الطول (padding/truncation)
        if len(sequence) < self.max_timesteps:
            pad_len = self.max_timesteps - len(sequence)
            sequence.extend([[0]*len(sequence[0])] * pad_len)
        else:
            sequence = sequence[:self.max_timesteps]
        
        return np.array(sequence)
