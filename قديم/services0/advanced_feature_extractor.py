from typing import List, Dict, Any
import numpy as np
import csv
import matplotlib.pyplot as plt

class AdvancedFeatureExtractor:
    """
    Extracts advanced gesture features:
    - spatial (mean, std, min, max)
    - velocity, acceleration
    - cumulative distance, radial distance
    - stroke progress & length
    """
    def __init__(self, max_timesteps:int=200, verbose:bool=False):
        self.max_timesteps = max_timesteps
        self.verbose = verbose
        self.per_frame_names = [
            "x_mean","x_std","y_mean","y_std","x_min","x_max","y_min","y_max",
            "vx","vy","ax","ay","cumulative_distance","stroke_progress","stroke_length",
            "delta_s","radial_distance"
        ]

    def _extract_frame_features(self, frame, prev_frame, prev_velocity, cumulative_dist, centroid):
        pts = frame.get("points",[]) or []
        delta_s = max(frame.get("delta_ms",1)/1000.0, 1e-6)

        if not pts:
            return {name: 0.0 for name in self.per_frame_names}

        x = np.array([p["x"] for p in pts], dtype=np.float32)
        y = np.array([p["y"] for p in pts], dtype=np.float32)
        x_mean, x_std = float(np.mean(x)), float(np.std(x))
        y_mean, y_std = float(np.mean(y)), float(np.std(y))
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))

        # السرعة
        if prev_frame:
            prev_pts = prev_frame.get("points", [])
            prev_x = np.mean([p["x"] for p in prev_pts]) if prev_pts else x_mean
            prev_y = np.mean([p["y"] for p in prev_pts]) if prev_pts else y_mean
            vx = (x_mean - prev_x)/delta_s
            vy = (y_mean - prev_y)/delta_s
        else:
            prev_x, prev_y = x_mean, y_mean
            vx, vy = 0.0, 0.0

        # التسارع
        if prev_velocity is not None:
            ax = (vx - prev_velocity[0])/delta_s
            ay = (vy - prev_velocity[1])/delta_s
        else:
            ax, ay = 0.0, 0.0

        # المسافات
        dist = np.sqrt((x_mean - prev_x)**2 + (y_mean - prev_y)**2) if prev_frame else 0.0
        cumulative_dist += dist
        stroke_length = cumulative_dist
        stroke_progress = cumulative_dist

        cx, cy = centroid
        radial_distance = float(np.sqrt((x_mean - cx)**2 + (y_mean - cy)**2))

        return {
            "x_mean":x_mean, "x_std":x_std, "y_mean":y_mean, "y_std":y_std,
            "x_min":x_min, "x_max":x_max, "y_min":y_min, "y_max":y_max,
            "vx":vx, "vy":vy, "ax":ax, "ay":ay,
            "cumulative_distance":cumulative_dist,
            "stroke_progress":stroke_progress,
            "stroke_length":stroke_length,
            "delta_s":delta_s,
            "radial_distance":radial_distance
        }

    def _gesture_to_sequence(self, gesture):
        frames = gesture.get("frames",[]) or []
        all_x = [p["x"] for f in frames for p in f.get("points",[])]
        all_y = [p["y"] for f in frames for p in f.get("points",[])]
        centroid = np.array([float(np.mean(all_x)) if all_x else 0.0,
                             float(np.mean(all_y)) if all_y else 0.0], dtype=np.float32)

        seq_features = []
        prev_velocity = None
        cumulative_dist = 0.0

        for i, frame in enumerate(frames):
            prev_frame = frames[i-1] if i>0 else None
            feat_dict = self._extract_frame_features(frame, prev_frame, prev_velocity, cumulative_dist, centroid)
            prev_velocity = np.array([feat_dict["vx"], feat_dict["vy"]], dtype=np.float32)
            cumulative_dist = feat_dict["cumulative_distance"]
            seq_features.append([feat_dict[name] for name in self.per_frame_names])

        seq_features = np.array(seq_features, dtype=np.float32)
        T,D = seq_features.shape

        # padding / resampling
        if T >= self.max_timesteps:
            idx = np.linspace(0, T-1, self.max_timesteps).astype(int)
            seq_features_fixed = seq_features[idx]
        else:
            pad_needed = self.max_timesteps - T
            seq_features_fixed = np.vstack([seq_features, np.zeros((pad_needed,D),dtype=np.float32)])

        return seq_features_fixed

    def save_gestures_to_csv(self, gestures: List[Dict[str,Any]], out_csv:str="ai_model/ADVANCED_features.csv"):
        if not gestures:
            print("❌ No gestures to process.")
            return
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # header
            header = ["gesture_id","character"] + self.per_frame_names
            writer.writerow(header)
            for g in gestures:
                seq = self._gesture_to_sequence(g)
                mean_features = np.mean(seq, axis=0).tolist()
                writer.writerow([g.get("id",0), g.get("character")] + mean_features)
        print(f"✅ Saved advanced features to {out_csv}")

    def plot_feature_variance(self, gestures: List[Dict[str,Any]], top_n: int = None):
        """
        تحسب التباين لكل ميزة وترسمها في رسم بياني واحد
        """
        if not gestures:
            print("❌ No gestures to process for plotting.")
            return

        # اجمع كل seq_features لكل gesture
        all_features = []
        for g in gestures:
            seq = self._gesture_to_sequence(g)
            mean_feat = np.mean(seq, axis=0)
            all_features.append(mean_feat)
        all_features = np.array(all_features)

        feature_variance = np.var(all_features, axis=0)
        feature_names = self.per_frame_names

        if top_n is not None and top_n < len(feature_names):
            idx = np.argsort(feature_variance)[-top_n:]
            feature_variance = feature_variance[idx]
            feature_names = [feature_names[i] for i in idx]

        plt.figure(figsize=(12,6))
        plt.bar(feature_names, feature_variance, color="skyblue")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Feature Variance")
        plt.title("Feature Variance Across Gestures (Low variance = possible noise)")
        plt.tight_layout()
        plt.show()
