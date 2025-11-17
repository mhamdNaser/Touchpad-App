from typing import List, Dict
import numpy as np
import pandas as pd


class AdvancedFeatureExtractor:
    """
    Extracts advanced gesture features including spatial, temporal,
    velocity, acceleration, and geometric features for each frame.
    """

    def __init__(self, max_timesteps: int = 200, verbose: bool = False):
        self.max_timesteps = max_timesteps
        self.verbose = verbose

        self.per_frame_names = [
            "x_mean", "x_std", "y_mean", "y_std",
            "x_min", "x_max", "y_min", "y_max",
            "pressure_mean",
            "vx", "vy", "ax", "ay",
            "angle", "curvature",
            "cumulative_distance",
            "stroke_progress",
            "stroke_length",
            "delta_s"
        ]

    # ---------------- extract features from a single frame ----------------
    def _extract_frame_features(self, frame: Dict, prev_frame: Dict = None, prev_velocity: np.ndarray = None, cumulative_dist: float = 0.0) -> Dict:
        pts = frame.get("points", []) or []
        delta_ms = frame.get("delta_ms", None)
        delta_s = (max(delta_ms, 1) / 1000.0) if delta_ms is not None else 0.0

        if not pts:
            return {name: 0.0 for name in self.per_frame_names}

        x = np.array([p.get("x", 0.0) for p in pts], dtype=np.float32)
        y = np.array([p.get("y", 0.0) for p in pts], dtype=np.float32)
        pressure = np.array([p.get("pressure", 0.0) for p in pts], dtype=np.float32)
        angle_pts = np.array([p.get("angle", 0.0) for p in pts], dtype=np.float32)

        # basic stats
        x_mean, x_std = float(np.mean(x)), float(np.std(x))
        y_mean, y_std = float(np.mean(y)), float(np.std(y))
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        pressure_mean = float(np.mean(pressure))

        # velocity
        if prev_frame is not None:
            prev_x = np.mean([p.get("x", 0.0) for p in prev_frame.get("points", [])] or [0.0])
            prev_y = np.mean([p.get("y", 0.0) for p in prev_frame.get("points", [])] or [0.0])
            vx = (x_mean - prev_x) / max(delta_s, 1e-6)
            vy = (y_mean - prev_y) / max(delta_s, 1e-6)
        else:
            vx, vy = 0.0, 0.0

        # acceleration
        if prev_velocity is not None:
            ax = (vx - prev_velocity[0]) / max(delta_s, 1e-6)
            ay = (vy - prev_velocity[1]) / max(delta_s, 1e-6)
        else:
            ax, ay = 0.0, 0.0

        # cumulative distance
        if prev_frame is not None:
            dist = np.sqrt((x_mean - prev_x)**2 + (y_mean - prev_y)**2)
        else:
            dist = 0.0
        cumulative_dist += dist

        # stroke progress & length
        stroke_length = cumulative_dist
        stroke_progress = cumulative_dist  # can be normalized later in sequence

        # curvature (simplified 2D curvature for 3 points if available)
        curvature = 0.0
        if prev_frame is not None:
            prev_prev_x = np.mean([p.get("x", 0.0) for p in prev_frame.get("prev_points", [])] or [prev_x])
            prev_prev_y = np.mean([p.get("y", 0.0) for p in prev_frame.get("prev_points", [])] or [prev_y])
            dx1, dy1 = prev_x - prev_prev_x, prev_y - prev_prev_y
            dx2, dy2 = x_mean - prev_x, y_mean - prev_y
            cross = dx1 * dy2 - dy1 * dx2
            dot = dx1 * dx2 + dy1 * dy2
            norm1 = np.sqrt(dx1**2 + dy1**2)
            norm2 = np.sqrt(dx2**2 + dy2**2)
            if norm1 * norm2 > 1e-6:
                curvature = float(cross / (norm1 * norm2))

        return {
            "x_mean": x_mean, "x_std": x_std,
            "y_mean": y_mean, "y_std": y_std,
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "pressure_mean": pressure_mean,
            "vx": vx, "vy": vy,
            "ax": ax, "ay": ay,
            "angle": float(np.mean(angle_pts)),
            "curvature": curvature,
            "cumulative_distance": cumulative_dist,
            "stroke_progress": stroke_progress,
            "stroke_length": stroke_length,
            "delta_s": delta_s
        }

    # ---------------- convert gesture → fixed-length array ----------------
    def _gesture_to_sequence(self, gesture: Dict) -> np.ndarray:
        frames = gesture.get("frames", []) or []
        seq = []
        prev_velocity = None
        cumulative_dist = 0.0

        for i, frame in enumerate(frames):
            prev_frame = frames[i - 1] if i > 0 else None
            feat_dict = self._extract_frame_features(frame, prev_frame, prev_velocity, cumulative_dist)
            prev_velocity = np.array([feat_dict["vx"], feat_dict["vy"]], dtype=np.float32)
            cumulative_dist = feat_dict["cumulative_distance"]
            seq.append([feat_dict[name] for name in self.per_frame_names])

        seq_arr = np.array(seq, dtype=np.float32)
        T = seq_arr.shape[0]

        if T >= self.max_timesteps:
            idx = np.linspace(0, T - 1, self.max_timesteps).astype(int)
            seq_fixed = seq_arr[idx]
        else:
            pad_needed = self.max_timesteps - T
            pad_block = np.zeros((pad_needed, len(self.per_frame_names)), dtype=np.float32)
            seq_fixed = np.vstack([seq_arr, pad_block])

        return seq_fixed

    # ---------------- save to CSV ----------------
    def process_gestures(self, gestures: List[Dict], out_csv: str = "gesture_features.csv") -> str:
        rows = []
        cols = ["gesture_id", "character", "orig_frame_count"] + self.per_frame_names * self.max_timesteps

        for g in gestures:
            gid = g.get("gesture_id") or g.get("id")
            char = g.get("character")
            orig_count = len(g.get("frames", []) or [])
            seq_fixed = self._gesture_to_sequence(g)
            row = [gid, char, orig_count] + seq_fixed.flatten().tolist()
            rows.append(row)

        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        return out_csv
