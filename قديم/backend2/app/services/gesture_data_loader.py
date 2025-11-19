#  app/services/gesture_data_loader.py
from typing import List, Dict, Optional, Tuple
import numpy as np
import requests

class GestureDataLoader:
    def __init__(self, api_url: str = "https://api.sydev.site/api/gestures", per_page: int = 50,
                 target_frames: int = 60, target_points: int = 20,
                 rotate_normalize: bool = True, center_to_zero: bool = True):
        self.api_url = api_url
        self.per_page = per_page
        self.target_frames = target_frames
        self.target_points = target_points
        self.session = requests.Session()
        # تكوين timeout كخاصية في الsession
        self.session.headers.update({"User-Agent": "gesture-loader/1.0"})
        self.timeout = 30
        self.allowed_states = ("down", "move")
        self.rotate_normalize = rotate_normalize
        self.center_to_zero = center_to_zero
        self.global_stats = None

    def _frame_sort_key(self, frame: Dict):
        return frame.get("timestamp") or frame.get("ts") or frame.get("frame_id") or 0

    def _collect_all_points(self, frames: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for f in frames:
            for p in f.get("points", []):
                xs.append(p["x"])
                ys.append(p["y"])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    def _compute_pca_angle(self, xs: np.ndarray, ys: np.ndarray) -> float:
        pts = np.vstack([xs, ys]).T
        if pts.shape[0] < 2:
            return 0.0
        pts_centered = pts - pts.mean(axis=0)
        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, np.argmax(eigvals)]
        angle = np.arctan2(principal[1], principal[0])
        return angle

    def compute_global_stats(self, all_gestures: List[Dict]) -> Dict:
        all_xs, all_ys, all_pressures = [], [], []
        for gesture in all_gestures:
            frames = gesture.get("frames", [])
            for frame in frames:
                for point in frame.get("points", []):
                    all_xs.append(point["x"])
                    all_ys.append(point["y"])
                    all_pressures.append(point.get("pressure", 1.0))
        if not all_xs:
            return {"x_mean": 0.0, "x_std": 1.0, "y_mean": 0.0, "y_std": 1.0, "pressure_mean": 1.0, "pressure_std": 0.1}
        stats = {
            "x_mean": float(np.mean(all_xs)),
            "x_std": max(float(np.std(all_xs)), 1e-6),
            "y_mean": float(np.mean(all_ys)),
            "y_std": max(float(np.std(all_ys)), 1e-6),
            "pressure_mean": float(np.mean(all_pressures)),
            "pressure_std": max(float(np.std(all_pressures)), 1e-6)
        }
        return stats

    def normalize_positions_global(self, frames: List[Dict], stats: Dict) -> List[Dict]:
        if not frames:
            return frames
        for frame in frames:
            for point in frame["points"]:
                point["x"] = (point["x"] - stats["x_mean"]) / stats["x_std"]
                point["y"] = (point["y"] - stats["y_mean"]) / stats["y_std"]
                point["pressure"] = (point.get("pressure", 1.0) - stats["pressure_mean"]) / stats["pressure_std"]
        return frames

    def normalize_positions_local(self, frames: List[Dict]) -> Dict[str, any]:
        if not frames:
            return {"frames": frames, "centroid": (0,0), "scale": 1.0, "rotation": 0.0}
        xs, ys = self._collect_all_points(frames)
        if xs.size == 0 or ys.size == 0:
            return {"frames": frames, "centroid": (0,0), "scale": 1.0, "rotation": 0.0}
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        for f in frames:
            for p in f.get("points", []):
                p["x"] = float(p["x"]) - cx
                p["y"] = float(p["y"]) - cy
        xs_c, ys_c = self._collect_all_points(frames)
        range_x = xs_c.max() - xs_c.min() if xs_c.size > 0 else 1.0
        range_y = ys_c.max() - ys_c.min() if ys_c.size > 0 else 1.0
        max_range = max(range_x, range_y, 1e-6)
        rotation = 0.0
        if self.rotate_normalize:
            angle = self._compute_pca_angle(xs_c, ys_c)
            rotation = -angle
            cos_a, sin_a = np.cos(rotation), np.sin(rotation)
            for f in frames:
                for p in f.get("points", []):
                    x, y = p["x"], p["y"]
                    rx = x * cos_a - y * sin_a
                    ry = x * sin_a + y * cos_a
                    p["x"], p["y"] = float(rx), float(ry)
        scale = 1.0 / max_range
        for f in frames:
            for p in f.get("points", []):
                p["x"] = float(p["x"]) * scale
                p["y"] = float(p["y"]) * scale
        return {"frames": frames, "centroid": (cx, cy), "scale": scale, "rotation": rotation}

    def resample_frames(self, frames: List[Dict], target_frames: Optional[int] = None) -> List[Dict]:
        target_frames = target_frames or self.target_frames
        if len(frames) <= 1:
            # ملء بالإطارات المتطابقة إن احتجنا لطول ثابت
            if not frames:
                return []
            f = frames[0]
            return [f for _ in range(target_frames)]
        timestamps = []
        for f in frames:
            ts = f.get("timestamp")
            if ts is None:
                if not timestamps:
                    timestamps.append(0.0)
                else:
                    timestamps.append(timestamps[-1] + f.get("delta_ms", 16) / 1000.0)
            else:
                timestamps.append(float(ts))
        timestamps = np.array(timestamps)
        t0, tN = timestamps[0], timestamps[-1]
        duration = max(tN - t0, 1e-6)
        original = (timestamps - t0) / duration
        target = np.linspace(0.0, 1.0, target_frames)
        max_points = max(len(f.get("points", [])) for f in frames) if frames else 1
        if max_points == 0:
            max_points = 1
        temp_frames = self.resample_points_per_frame(frames, target_points=max_points)
        resampled = []
        for t in target:
            idx = np.searchsorted(original, t)
            if idx == 0:
                f_interp = temp_frames[0]
            elif idx >= len(original):
                f_interp = temp_frames[-1]
            else:
                t0_ = original[idx-1]
                t1_ = original[idx]
                alpha = 0.0 if abs(t1_ - t0_) < 1e-9 else (t - t0_) / (t1_ - t0_)
                f1, f2 = temp_frames[idx-1], temp_frames[idx]
                interpolated_points = []
                for p1, p2 in zip(f1["points"], f2["points"]):
                    x = (1 - alpha) * p1["x"] + alpha * p2["x"]
                    y = (1 - alpha) * p1["y"] + alpha * p2["y"]
                    pressure = (1 - alpha) * p1.get("pressure", 1.0) + alpha * p2.get("pressure", 1.0)
                    interpolated_points.append({"x": float(x), "y": float(y), "pressure": float(pressure)})
                ts_interp = int((1 - alpha) * f1.get("timestamp", 0) + alpha * f2.get("timestamp", 0))
                delta_interp = int((1 - alpha) * f1.get("delta_ms", 0) + alpha * f2.get("delta_ms", 0))
                resampled.append({"timestamp": ts_interp, "delta_ms": delta_interp, "points": interpolated_points})
                continue
            copied_points = []
            for p in f_interp["points"]:
                copied_points.append({"x": float(p["x"]), "y": float(p["y"]), "pressure": float(p.get("pressure", 1.0))})
            resampled.append({"timestamp": f_interp.get("timestamp", 0), "delta_ms": f_interp.get("delta_ms", 0), "points": copied_points})
        return resampled

    def resample_points_per_frame(self, frames: List[Dict], target_points: Optional[int] = None) -> List[Dict]:
        target_points = target_points or self.target_points
        new_frames = []
        for f in frames:
            pts = f.get("points", [])
            if len(pts) == 0:
                new_frames.append({**f, "points": [{"x": 0.0, "y": 0.0, "pressure": 1.0} for _ in range(target_points)]})
                continue
            if len(pts) == target_points:
                new_pts = []
                for p in pts:
                    new_pts.append({"x": float(p["x"]), "y": float(p["y"]), "pressure": float(p.get("pressure", 1.0))})
                new_frames.append({**f, "points": new_pts})
                continue
            xs = np.array([float(p["x"]) for p in pts])
            ys = np.array([float(p["y"]) for p in pts])
            pressures = np.array([float(p.get("pressure", 1.0)) for p in pts])
            dx = np.diff(xs, prepend=xs[0])
            dy = np.diff(ys, prepend=ys[0])
            dists = np.sqrt(dx**2 + dy**2)
            cum_dist = np.cumsum(dists)
            total_dist = cum_dist[-1]
            if total_dist == 0:
                original_param = np.linspace(0, 1, len(pts))
            else:
                original_param = cum_dist / total_dist
            target_param = np.linspace(0.0, 1.0, target_points)
            new_xs = np.interp(target_param, original_param, xs)
            new_ys = np.interp(target_param, original_param, ys)
            new_pressures = np.interp(target_param, original_param, pressures)
            new_pts = []
            for i in range(target_points):
                new_pts.append({"x": float(new_xs[i]), "y": float(new_ys[i]), "pressure": float(new_pressures[i])})
            new_frames.append({**f, "points": new_pts})
        return new_frames

    def _process_gesture(self, gesture: Dict, global_stats: Dict = None) -> Optional[Dict]:
        frames_raw = gesture.get("frames", [])
        points_raw = gesture.get("points", [])
        frames = []
        if frames_raw:
            sorted_frames = sorted(frames_raw, key=self._frame_sort_key)
            prev_ts = None
            for f in sorted_frames:
                ts = f.get("timestamp") or f.get("ts") or 0
                delta = f.get("delta_ms", 0)
                if not delta and prev_ts is not None:
                    delta = ts - prev_ts
                prev_ts = ts
                pts = [{"x": p.get("x", 0.0), "y": p.get("y", 0.0), "pressure": p.get("pressure", 1.0)} for p in f.get("points", []) if p.get("state") in self.allowed_states]
                if pts:
                    frames.append({"timestamp": ts, "delta_ms": delta, "points": pts})
        elif points_raw:
            temp = {}
            for p in points_raw:
                fid = p.get("frame_id")
                if fid not in temp:
                    temp[fid] = {"timestamp": p.get("timestamp", 0), "delta_ms": p.get("delta_ms", 0), "points": []}
                if p.get("state") in self.allowed_states:
                    temp[fid]["points"].append({"x": p["x"], "y": p["y"], "pressure": p.get("pressure", 1.0)})
            frames = sorted([f for f in temp.values() if f["points"]], key=self._frame_sort_key)
        if not frames:
            return None
        frames = self.resample_frames(frames, self.target_frames)
        frames = self.resample_points_per_frame(frames, self.target_points)
        if global_stats:
            frames = self.normalize_positions_global(frames, global_stats)
        prev_points = None
        for frame in frames:
            current_points = frame["points"]
            for i, point in enumerate(current_points):
                if prev_points is None or i >= len(prev_points):
                    point["dx"] = 0.0
                    point["dy"] = 0.0
                    point["angle"] = 0.0
                else:
                    prev_point = prev_points[i]
                    dx = point["x"] - prev_point["x"]
                    dy = point["y"] - prev_point["y"]
                    point["dx"] = dx
                    point["dy"] = dy
                    point["angle"] = float(np.arctan2(dy, dx)) if (dx != 0 or dy != 0) else 0.0
            prev_points = current_points
        return {"gesture_id": gesture.get("id"), "character": gesture.get("character"), "duration_ms": gesture.get("duration_ms", 0), "frames": frames, "frame_count": len(frames), "point_count": self.target_points}

    def load_all_gestures(self) -> List[Dict]:
        page = 1
        all_raw_gestures = []
        print("📥 جاري تحميل البيانات الخام...")
        while True:
            url = f"{self.api_url}?page={page}&per_page={self.per_page}"
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                gestures = data.get("data", []) if isinstance(data, dict) else data
                if not gestures:
                    break
                all_raw_gestures.extend(gestures)
                print(f"📄 تم تحميل الصفحة {page} - {len(gestures)} إيماءة")
                page += 1
            except Exception as e:
                print(f"❌ خطأ في تحميل الصفحة {page}: {e}")
                break
        if not all_raw_gestures:
            print("❌ لم يتم تحميل أي إيماءات")
            return []
        print(f"📊 إجمالي الإيماءات الخام: {len(all_raw_gestures)}")
        print("📈 جاري حساب الإحصائيات العالمية...")
        self.global_stats = self.compute_global_stats(all_raw_gestures)
        print(f"🎯 الإحصائيات العالمية: {self.global_stats}")
        processed_gestures = []
        successful = 0
        for i, raw_gesture in enumerate(all_raw_gestures):
            try:
                processed = self._process_gesture(raw_gesture, self.global_stats)
                if processed:
                    processed_gestures.append(processed)
                    successful += 1
            except Exception as e:
                print(f"⚠️ خطأ في معالجة الإيماءة {raw_gesture.get('id')}: {e}")
                continue
            if (i + 1) % 50 == 0:
                print(f"✅ تم معالجة {i + 1}/{len(all_raw_gestures)} إيماءة")
        print(f"🎉 تم معالجة {successful}/{len(all_raw_gestures)} إيماءة بنجاح")
        output = []
        for g in processed_gestures:
            tensor = self.gesture_to_tensor(g)
            output.append({"X": tensor, "y": g["character"], "id": g["gesture_id"], "raw": g})
        return output

    def gesture_to_tensor(self, gesture: Dict) -> np.ndarray:
        frames = gesture["frames"]
        F = len(frames)
        P = self.target_points
        features = 6
        tensor = np.zeros((F, P, features), dtype=np.float32)
        for fi, frame in enumerate(frames):
            for pi, point in enumerate(frame["points"]):
                tensor[fi, pi, 0] = float(point.get("x", 0.0))
                tensor[fi, pi, 1] = float(point.get("y", 0.0))
                tensor[fi, pi, 2] = float(point.get("dx", 0.0))
                tensor[fi, pi, 3] = float(point.get("dy", 0.0))
                tensor[fi, pi, 4] = float(point.get("angle", 0.0))
                tensor[fi, pi, 5] = float(point.get("pressure", 1.0))
        return tensor

    def get_global_stats(self) -> Dict:
        return self.global_stats
