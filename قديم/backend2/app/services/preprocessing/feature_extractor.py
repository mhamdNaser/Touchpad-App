# app/services/preprocessing/feature_extractor.py
import numpy as np
import torch

class FeatureExtractor:
    def __init__(self):
        pass

    def compute_delta(self, gesture):
        diff = np.diff(gesture, axis=0)
        diff = np.vstack([diff[0:1], diff])
        return diff

    def compute_angle(self, gesture):
        dxdy = self.compute_delta(gesture)
        dx = dxdy[..., 0]
        dy = dxdy[..., 1]
        angle = np.arctan2(dy, dx)
        return angle[..., None]

    def extract(self, gesture):
        delta = self.compute_delta(gesture)
        angle = self.compute_angle(gesture)
        features = np.concatenate([gesture, delta, angle], axis=-1)
        return features

    def to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float32)
