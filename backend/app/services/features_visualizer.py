
"""
Production-ready feature extractor for per-frame features suitable for sequence models (RNN/CNN-1D/Transformer).

Design decisions (based on your stats and our discussion):
- We keep the 11 robust features per frame:
  [x_mean, x_std, y_mean, y_std, x_min, x_max, y_min, y_max, pressure_mean, angle_mean, delta_s]
- Output default: a "wide" CSV (one row per gesture) where per-frame features are flattened and padded/truncated to `max_timesteps`.
- Also supports optional "long" CSV (one row per frame) if you prefer working with long format.
- Minimal prints, production safe (error handling, dtype control).
- Companion metadata (gesture_id, character, original_frame_count) included.

Usage:
    extractor = ProductionFeatureExtractor(loader=None, max_timesteps=150)
    # loader should be an object with .load_all_gestures() OR pass a list of gestures to process_gestures
    gestures = loader.load_all_gestures()
    extractor.process_gestures(gestures, out_csv="gesture_features_wide.csv")

You can later load the CSV and reshape each row to (max_timesteps, feature_dim) for training sequence models.
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd


class ProductionFeatureExtractor:
    def __init__(self, max_timesteps: int = 150, feature_dim: int = 11, verbose: bool = False):
        self.max_timesteps = max_timesteps
        self.feature_dim = feature_dim  # must be 11 (kept for clarity)
        self.verbose = verbose
        # Feature names per-frame
        self.per_frame_names = [
            "x_mean", "x_std",
            "y_mean", "y_std",
            "x_min", "x_max",
            "y_min", "y_max",
            "pressure_mean", "angle_mean",
            "delta_s"
        ]

    # ---------------- core: extract features from a single frame ----------------
    def _extract_frame_features(self, frame: Dict) -> List[float]:
        pts = frame.get("points", []) or []

        # timestamp/delta handling
        delta_ms = frame.get("delta_ms", None)
        # delta_s default (in seconds)
        delta_s = (max(delta_ms, 1) / 1000.0) if delta_ms is not None else 0.016

        if not pts:
            # Return zeros except delta_s
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(delta_s)]

        # gather arrays
        x = np.array([p.get("x", 0.0) for p in pts], dtype=np.float32)
        y = np.array([p.get("y", 0.0) for p in pts], dtype=np.float32)
        pressure = np.array([p.get("pressure", 0.0) for p in pts], dtype=np.float32)
        angle = np.array([p.get("angle", 0.0) for p in pts], dtype=np.float32)

        # safe stats
        x_mean = float(np.mean(x)) if x.size else 0.0
        x_std = float(np.std(x)) if x.size else 0.0
        y_mean = float(np.mean(y)) if y.size else 0.0
        y_std = float(np.std(y)) if y.size else 0.0
        x_min = float(np.min(x)) if x.size else 0.0
        x_max = float(np.max(x)) if x.size else 0.0
        y_min = float(np.min(y)) if y.size else 0.0
        y_max = float(np.max(y)) if y.size else 0.0
        pressure_mean = float(np.mean(pressure)) if pressure.size else 0.0
        angle_mean = float(np.mean(angle)) if angle.size else 0.0

        return [
            x_mean, x_std,
            y_mean, y_std,
            x_min, x_max,
            y_min, y_max,
            pressure_mean, angle_mean,
            float(delta_s)
        ]

    # ---------------- extract per-gesture: sequence of frames -> fixed-length array ----------------
    def _gesture_to_sequence(self, gesture: Dict) -> np.ndarray:
        frames = gesture.get("frames", []) or []
        seq = []

        for frame in frames:
            f = self._extract_frame_features(frame)
            seq.append(f)

        # convert to numpy array (T, D)
        if len(seq) == 0:
            seq_arr = np.zeros((0, self.feature_dim), dtype=np.float32)
        else:
            seq_arr = np.array(seq, dtype=np.float32)

        T = seq_arr.shape[0]

        if T >= self.max_timesteps:
            # downsample uniformly to max_timesteps to preserve temporal structure
            idx = np.linspace(0, T - 1, self.max_timesteps).astype(int)
            seq_fixed = seq_arr[idx]
        else:
            # pad at the end with zeros (preserve temporal order) — keep delta_s for padding as last seen delta
            pad_needed = self.max_timesteps - T
            if T == 0:
                seq_fixed = np.zeros((self.max_timesteps, self.feature_dim), dtype=np.float32)
            else:
                last_delta = seq_arr[-1, -1]
                pad_block = np.zeros((pad_needed, self.feature_dim), dtype=np.float32)
                pad_block[:, -1] = last_delta  # keep delta_s for padding
                seq_fixed = np.vstack([seq_arr, pad_block])

        return seq_fixed

    # ---------------- public: process gestures list and save CSV ----------------
    def process_gestures(self, gestures: List[Dict], out_csv: str = "gesture_features_wide.csv", format: str = "wide") -> str:
        """
        gestures: list of gesture dicts
        out_csv: output filename
        format: 'wide' (one row per gesture, flattened) or 'long' (one row per frame)
        Returns: path to saved CSV
        """
        rows = []

        if format not in ("wide", "long"):
            raise ValueError("format must be 'wide' or 'long'")

        if format == "wide":
            # prepare column names
            cols = ["gesture_id", "character", "orig_frame_count"]
            for t in range(self.max_timesteps):
                for fn in self.per_frame_names:
                    cols.append(f"t{t:03d}_{fn}")

            # iterate and build rows
            for g in gestures:
                gid = g.get("gesture_id") or g.get("id")
                char = g.get("character")
                orig_count = len(g.get("frames", []) or [])
                seq_fixed = self._gesture_to_sequence(g)  # shape (max_timesteps, feature_dim)
                flat = seq_fixed.flatten()
                row = [gid, char, orig_count] + flat.tolist()
                rows.append(row)

            df = pd.DataFrame(rows, columns=cols)
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")

            if self.verbose:
                print(f"Saved wide CSV: {out_csv} (rows={len(df)})")
            return out_csv

        else:
            # long format: one row per frame
            cols = ["gesture_id", "character", "frame_index", "orig_frame_count"] + self.per_frame_names
            for g in gestures:
                gid = g.get("gesture_id") or g.get("id")
                char = g.get("character")
                frames = g.get("frames", []) or []
                orig_count = len(frames)
                seq = self._gesture_to_sequence(g)[:orig_count] if orig_count > 0 else np.zeros((0, self.feature_dim))
                for i in range(seq.shape[0]):
                    row = [gid, char, int(i), orig_count] + seq[i].tolist()
                    rows.append(row)

            df = pd.DataFrame(rows, columns=cols)
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            if self.verbose:
                print(f"Saved long CSV: {out_csv} (rows={len(df)})")
            return out_csv


# ------------------ Utility function: reshape flattened row back to sequence ------------------
def reshape_flat_row(flat_row: List[float], max_timesteps: int = 150, feature_dim: int = 11) -> np.ndarray:
    arr = np.array(flat_row, dtype=np.float32)
    expected = max_timesteps * feature_dim
    if arr.size != expected:
        raise ValueError(f"Flat row length {arr.size} != expected {expected}")
    return arr.reshape((max_timesteps, feature_dim))


# ---------------- example usage ----------------
if __name__ == "__main__":
    # small example: expecting you have a loader with load_all_gestures()
    try:
        from app.services.gesture_data_loader import GestureDataLoader
        loader = GestureDataLoader()
        gestures = loader.load_all_gestures()
    except Exception:
        gestures = []

    extractor = ProductionFeatureExtractor(max_timesteps=150, verbose=True)
    if gestures:
        extractor.process_gestures(gestures, out_csv="gesture_features_wide.csv", format="wide")
    else:
        print("No gestures available to process in this environment.")