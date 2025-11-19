# app/services/preprocessing/normalization.py
import numpy as np

class Normalizer:
    def __init__(self, pca_rotation=True):
        self.pca_rotation = pca_rotation
        self.global_mean = None
        self.global_std = None

    def fit_global(self, all_samples):
        """all_samples shape: (N, frames, points*2)"""
        flat = all_samples.reshape(len(all_samples), -1)
        self.global_mean = flat.mean(axis=0)
        self.global_std = flat.std(axis=0) + 1e-6

    def apply_global(self, sample):
        flat = sample.reshape(-1)
        flat = (flat - self.global_mean) / self.global_std
        return flat.reshape(sample.shape)

    def center(self, gesture):
        mean = gesture.mean(axis=(0,1))
        return gesture - mean

    def apply_pca(self, gesture):
        pts = gesture.reshape(-1, 2)
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eig(cov)

        if eigvals.max() < 1e-4:
            return gesture

        rotation = eigvecs[:, np.argmax(eigvals)]
        angle = np.arctan2(rotation[1], rotation[0])
        R = np.array([[np.cos(-angle), -np.sin(-angle)],
                      [np.sin(-angle),  np.cos(-angle)]])
        rotated = pts @ R.T
        return rotated.reshape(gesture.shape)

    def normalize(self, gesture):
        gesture = self.center(gesture)
        if self.pca_rotation:
            gesture = self.apply_pca(gesture)
        return gesture
