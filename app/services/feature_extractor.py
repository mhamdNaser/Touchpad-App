# app/services/feature_extractor.py
import numpy as np

class GestureFeatureExtractor:
    def __init__(self):
        pass

    def compute_velocity(self, points):
        points = np.array(points)
        dx = np.diff(points[:,0])
        dy = np.diff(points[:,1])
        dt = 1  # أو يمكن استخدام delta_ms إذا متوفر
        vx = dx / dt
        vy = dy / dt
        return np.column_stack((vx, vy))

    def compute_angles(self, points):
        points = np.array(points)
        vectors = np.diff(points, axis=0)
        angles = np.arctan2(vectors[:,1], vectors[:,0])
        return angles

    def extract_features(self, gesture):
        coords = np.array([[p['x'], p['y']] for p in gesture])
        velocity = self.compute_velocity(coords)
        angles = self.compute_angles(coords)
        features = np.concatenate([coords.flatten(), velocity.flatten(), angles.flatten()])
        return features
