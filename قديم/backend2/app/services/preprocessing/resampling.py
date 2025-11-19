# app/services/preprocessing/resampling.py
import numpy as np

class Resampler:
    def __init__(self, target_frames=60, target_points=21):
        self.target_frames = target_frames
        self.target_points = target_points

    def resample_points(self, points):
        # points = (N, 2)
        idx = np.linspace(0, len(points) - 1, self.target_points)
        idx_floor = np.floor(idx).astype(int)
        idx_ceil = np.ceil(idx).astype(int)
        alpha = idx - idx_floor
        resampled = (1 - alpha)[:, None] * points[idx_floor] + alpha[:, None] * points[idx_ceil]
        return resampled

    def resample_frames(self, frames):
        processed = []
        for frame in frames:
            pts = np.array([[p["x"], p["y"]] for p in frame])
            pts = self.resample_points(pts)
            processed.append(pts)

        processed = np.array(processed)
        time_idx = np.linspace(0, len(processed) - 1, self.target_frames)
        time_floor = np.floor(time_idx).astype(int)
        time_ceil = np.ceil(time_idx).astype(int)
        alpha = time_idx - time_floor

        result = (1 - alpha)[:, None, None] * processed[time_floor] + alpha[:, None, None] * processed[time_ceil]
        return result