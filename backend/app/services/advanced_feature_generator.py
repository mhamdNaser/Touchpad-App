from typing import List, Dict
import numpy as np
import pandas as pd


class StatisticalFeatureGenerator:
    """
    نسخة مبسّطة للإحصاء فقط
    مناسبة عندما يكون كل Frame يحتوي نقطة واحدة فقط.
    """

    def __init__(self, max_timesteps: int = 200, verbose: bool = False):
        self.max_timesteps = max_timesteps
        self.verbose = verbose

        # الفيتشرات مفيدة لفريم من نقطة واحدة فقط
        self.per_frame_names = [
            "x",
            "y",
            "vx",
            "vy",
            "angle",
            "pressure",
            "active_points",
            "delta_s"
        ]

        self.feature_dim = len(self.per_frame_names)

    # ---------------------------------------------------------------------
    # استخراج فيتشر فريم واحد
    # ---------------------------------------------------------------------
    def _extract_frame(self, frame: Dict) -> List[float]:
        pts = frame.get("points", []) or []
        delta_ms = frame.get("delta_ms", 16)
        delta_s = max(delta_ms, 1) / 1000.0

        if not pts:
            # فريم فارغ (padding)
            return [0, 0, 0, 0, 0, 0, 0, delta_s]

        p = pts[0]  # نقطة واحدة فقط

        return [
            float(p.get("x", 0)),
            float(p.get("y", 0)),
            float(p.get("vx", 0)),
            float(p.get("vy", 0)),
            float(p.get("angle", 0)),
            float(p.get("pressure", 0)),
            1.0,            # always 1 point
            delta_s
        ]

    # ---------------------------------------------------------------------
    # تحويل الجستشر إلى سيكوينس ثابت (downsample + padding)
    # ---------------------------------------------------------------------
    def _gesture_to_vector(self, gesture: Dict) -> np.ndarray:
        frames = gesture.get("frames", []) or []
        seq = [self._extract_frame(f) for f in frames]

        seq_arr = np.array(seq, dtype=np.float32) if len(seq) else np.zeros((0, self.feature_dim), dtype=np.float32)
        T = seq_arr.shape[0]

        # Downsample
        if T >= self.max_timesteps:
            idx = np.linspace(0, T - 1, self.max_timesteps).astype(int)
            return seq_arr[idx]

        # Padding
        pad = self.max_timesteps - T
        if pad > 0:
            pad_block = np.zeros((pad, self.feature_dim), dtype=np.float32)
            # احتفظ بآخر delta_s حتى لا تضيع حركة الزمن
            last_delta = seq_arr[-1, -1] if T > 0 else 0.016
            pad_block[:, -1] = last_delta
            return np.vstack([seq_arr, pad_block])

        return seq_arr

    # ---------------------------------------------------------------------
    # الدالة العامة: انتاج CSV إحصائي فقط
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

                for i, fvec in enumerate(seq):
                    rows.append([gid, char, i] + fvec.tolist())

            df = pd.DataFrame(rows, columns=cols)
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")

            if self.verbose:
                print(f"Saved LONG CSV → {out_csv}")

        return out_csv
