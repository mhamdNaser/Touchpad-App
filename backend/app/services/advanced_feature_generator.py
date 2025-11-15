from typing import List, Dict
import numpy as np
import pandas as pd


class AdvancedFeatureGenerator:
    """
    نسخة محسّنة وموسّعة لاستخراج أكبر عدد من الفيتشرات الممكنة
    من البيانات الخام بدون أي علاقة بالكلاس القديم.
    """

    def __init__(self, max_timesteps: int = 50, verbose: bool = False):
        self.max_timesteps = max_timesteps
        self.verbose = verbose

        # 21 فيتشر لكل فريم
        self.per_frame_names = [
            # position stats
            "x_mean", "x_std", "x_min", "x_max",
            "y_mean", "y_std", "y_min", "y_max",

            # velocity stats
            "vx_mean", "vx_std", "vx_min", "vx_max",
            "vy_mean", "vy_std", "vy_min", "vy_max",

            # pressure stats
            "pressure_mean", "pressure_std", "pressure_min", "pressure_max",

            # angle stats
            "angle_mean", "angle_std", "angle_min", "angle_max",

            # extra
            "active_points",
            "delta_s"
        ]

        self.feature_dim = len(self.per_frame_names)

    # ---------------------------------------------------------------------
    # استخراج فيتشر لفريم واحد
    # ---------------------------------------------------------------------
    def _extract_frame(self, frame: Dict) -> List[float]:
        pts = frame.get("points", []) or []
        delta_ms = frame.get("delta_ms", 16)
        delta_s = max(delta_ms, 1) / 1000.0

        if not pts:
            return [0] * (self.feature_dim - 1) + [delta_s]

        x = np.array([p.get("x", 0) for p in pts])
        y = np.array([p.get("y", 0) for p in pts])
        vx = np.array([p.get("vx", 0) for p in pts])
        vy = np.array([p.get("vy", 0) for p in pts])
        angle = np.array([p.get("angle", 0) for p in pts])
        pressure = np.array([p.get("pressure", 0) for p in pts])

        features = [
            # x stats
            float(x.mean()), float(x.std()), float(x.min()), float(x.max()),
            # y stats
            float(y.mean()), float(y.std()), float(y.min()), float(y.max()),
            # velocity
            float(vx.mean()), float(vx.std()), float(vx.min()), float(vx.max()),
            float(vy.mean()), float(vy.std()), float(vy.min()), float(vy.max()),
            # pressure
            float(pressure.mean()), float(pressure.std()), float(pressure.min()), float(pressure.max()),
            # angle
            float(angle.mean()), float(angle.std()), float(angle.min()), float(angle.max()),
            # active points
            float(len(pts)),
            # delta
            float(delta_s),
        ]

        return features

    # ---------------------------------------------------------------------
    # تحويل الجستشر إلى سيكوينس ثابت
    # ---------------------------------------------------------------------
    def _gesture_to_vector(self, gesture: Dict) -> np.ndarray:
        frames = gesture.get("frames", []) or []
        seq = [self._extract_frame(f) for f in frames]

        if len(seq) == 0:
            seq_arr = np.zeros((0, self.feature_dim), dtype=np.float32)
        else:
            seq_arr = np.array(seq, dtype=np.float32)

        T = seq_arr.shape[0]

        # Downsample
        if T >= self.max_timesteps:
            idx = np.linspace(0, T - 1, self.max_timesteps).astype(int)
            return seq_arr[idx]

        # Pad
        pad = self.max_timesteps - T
        if pad > 0:
            last_delta = seq_arr[-1, -1] if T > 0 else 0.016
            pad_block = np.zeros((pad, self.feature_dim), dtype=np.float32)
            pad_block[:, -1] = last_delta
            return np.vstack([seq_arr, pad_block])

        return seq_arr

    # ---------------------------------------------------------------------
    # الدالة العامة — الاسم الجديد generate_features()
    # ---------------------------------------------------------------------
    def generate_features(self, gestures: List[Dict], out_csv: str, format: str = "wide"):
        rows = []

        if format == "wide":
            cols = ["gesture_id", "character", "orig_frames"]
            for t in range(self.max_timesteps):
                for fn in self.per_frame_names:
                    cols.append(f"t{t:03d}_{fn}")

            for g in gestures:
                gid = g.get("id")
                char = g.get("character")
                frames = g.get("frames", [])
                seq = self._gesture_to_vector(g).flatten().tolist()

                rows.append([gid, char, len(frames)] + seq)

            df = pd.DataFrame(rows, columns=cols)
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")

            if self.verbose:
                print(f"Saved WIDE CSV → {out_csv}")

        else:
            cols = ["gesture_id", "character", "frame_index"] + self.per_frame_names

            for g in gestures:
                gid = g.get("id")
                char = g.get("character")
                seq = self._gesture_to_vector(g)

                for i, frame_vec in enumerate(seq):
                    rows.append([gid, char, i] + frame_vec.tolist())

            df = pd.DataFrame(rows, columns=cols)
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")

            if self.verbose:
                print(f"Saved LONG CSV → {out_csv}")

        return out_csv
