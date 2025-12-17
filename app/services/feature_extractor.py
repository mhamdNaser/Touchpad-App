# app/services/feature_extractor.py
import numpy as np
from typing import List, Optional, Tuple, Union

class GestureFeatureExtractor:
    """
    يحول الإيماءة (قائمة نقاط أو مصفوفة [T,2]) إلى صورة ثابتة (H x W)
    - يدعم قنوات: stroke (المسار)، velocity (السرعة)، pressure (إذا متوفّر)
    - يمكن تصدير الفيتشر كـ flattened vector أو كـ HxWxC numpy array
    """

    def __init__(self,
                 image_size: int = 64,
                 thickness: float = 2.0,
                 channels: Tuple[str, ...] = ("stroke", "velocity"),  # خيارات: "stroke","velocity","pressure"
                 resample_frames: int = 100):
        self.image_size = image_size
        self.thickness = thickness
        self.channels = channels
        self.resample_frames = resample_frames

    # ---------- أدوات مساعدة ----------
    def _to_pixel_coords(self, coords_norm: np.ndarray) -> np.ndarray:
        """
        coords_norm: [T,2] بقيم بين 0 و 1
        ترجع إحداثيات بكسل كـ float [T,2] داخل مجال [0, image_size-1]
        """
        coords = np.clip(coords_norm, 0.0, 1.0)
        px = coords * (self.image_size - 1)
        return px

    def _compute_velocity_magnitudes(self, coords: np.ndarray) -> np.ndarray:
        """
        coords: [T,2]
        ترجع سرعة لكل إطار طول T (بالنسبة لإطار 0 تُعطى 0)
        """
        deltas = np.diff(coords, axis=0)
        speeds = np.linalg.norm(deltas, axis=1)
        speeds = np.concatenate([[0.0], speeds])  # frame 0 speed = 0
        # normalize to 0..1 to be image-intensity-friendly
        maxv = speeds.max() if speeds.max() > 0 else 1.0
        return speeds / maxv

    def _rasterize_line(self, canvas: np.ndarray, p0: Union[Tuple[float, float], np.ndarray], p1: Union[Tuple[float, float], np.ndarray], intensity: float):
        """
        بسيط: نمشي على خط بين p0 و p1 ونملأ نقاط على طول الخط
        canvas: 2D float array
        p0, p1: float pixel coords (tuple or np.ndarray)
        intensity: value to add (0..1)
        """
        p0 = np.array(p0)
        p1 = np.array(p1)
        dist = np.linalg.norm(p1 - p0)
        if dist == 0:
            ix = int(round(p0[1]))
            iy = int(round(p0[0]))
            if 0 <= ix < self.image_size and 0 <= iy < self.image_size:
                canvas[ix, iy] = max(canvas[ix, iy], intensity)
            return

        steps = int(max(2, np.ceil(dist * 2)))  # more samples for longer segments
        for t in np.linspace(0, 1, steps):
            x, y = p0 * (1-t) + p1 * t
            # round to nearest pixel indices (row,col) -> (y,x)
            col = int(round(x))
            row = int(round(y))
            # Spread intensity to neighbors according to thickness
            r = int(np.ceil(self.thickness))
            for rr in range(row - r, row + r + 1):
                for cc in range(col - r, col + r + 1):
                    if 0 <= rr < self.image_size and 0 <= cc < self.image_size:
                        # gaussian-ish dropoff by distance
                        d = np.hypot(rr - row, cc - col)
                        add = intensity * max(0.0, 1.0 - (d / (self.thickness + 1e-6)))
                        canvas[rr, cc] = max(canvas[rr, cc], add)

    # ---------- الواجهة العامة ----------
    def gesture_to_image(self,
                         gesture: Union[List[dict], np.ndarray],
                         normalize_coords: bool = True
                         ) -> np.ndarray:
        """
        gesture: إما قائمة نقاط [{'x','y','pressure'?}, ...] أو np.array [T,2] (قيم خام)
        ترجع: numpy array بحجم (H, W, C) حيث C = len(self.channels)
        القيم في كل قناة محصورة بين 0 و 1 (float32)
        """
        # 1) الحصول على مصفوفة coords [T,2] وأيضًا pressures إن وُجِدت
        if isinstance(gesture, np.ndarray):
            coords = gesture.copy()
            pressures = None
        else:
            coords = np.array([[p['x'], p['y']] for p in gesture], dtype=float)
            pressures = None
            if 'pressure' in gesture[0]:
                pressures = np.array([p.get('pressure', 0.0) for p in gesture], dtype=float)

        # 2) optional normalization to 0..1 using bbox
        if normalize_coords:
            minv = coords.min(axis=0)
            maxv = coords.max(axis=0)
            denom = (maxv - minv)
            denom[denom == 0] = 1.0
            coords = (coords - minv) / denom

        # 3) resample/interpolate to fixed frames for stability (self.resample_frames)
        t_src = np.linspace(0, 1, coords.shape[0])
        t_dst = np.linspace(0, 1, self.resample_frames)
        coords_r = np.vstack([
            np.interp(t_dst, t_src, coords[:, 0]),
            np.interp(t_dst, t_src, coords[:, 1])
        ]).T  # [resample_frames,2]
        if pressures is not None:
            pressures_r = np.interp(t_dst, t_src, pressures)
        else:
            pressures_r = None

        # 4) pixel coordinates
        px = self._to_pixel_coords(coords_r)  # [T,2] float (x,y) in pixel space (0..image_size-1)

        # 5) create empty channels
        H = W = self.image_size
        C = len(self.channels)
        img = np.zeros((H, W, C), dtype=np.float32)

        # 6) compute velocity channel if requested
        velocity = None
        if "velocity" in self.channels:
            velocity = self._compute_velocity_magnitudes(coords_r)  # normalized 0..1 length T

        # 7) rasterize strokes
        # We'll draw segment-by-segment; intensity might depend on velocity or pressure
        for i in range(len(px) - 1):
            p0 = px[i]
            p1 = px[i + 1]
            # base intensity for stroke channel
            stroke_intensity = 1.0
            if "velocity" in self.channels and velocity is not None:
                # weight stroke brightness by local velocity (or inverse) — here use velocity
                stroke_intensity *= (0.3 + 0.7 * velocity[i])  # keep some minimum
            if "pressure" in self.channels and pressures_r is not None:
                stroke_intensity *= (0.3 + 0.7 * pressures_r[i])

            # draw into stroke channel (index accordingly)
            if "stroke" in self.channels:
                idx = self.channels.index("stroke")
                self._rasterize_line(img[:, :, idx], (p0[0], p0[1]), (p1[0], p1[1]), stroke_intensity)

            # draw into velocity channel: set pixel value proportional to velocity
            if "velocity" in self.channels and velocity is not None:
                idx = self.channels.index("velocity")
                self._rasterize_line(img[:, :, idx], (p0[0], p0[1]), (p1[0], p1[1]), velocity[i])

            # draw pressure channel if present
            if "pressure" in self.channels and pressures_r is not None:
                idx = self.channels.index("pressure")
                self._rasterize_line(img[:, :, idx], (p0[0], p0[1]), (p1[0], p1[1]), pressures_r[i])

        # 8) final normalization: clamp 0..1
        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        return img

    def extract_features(self, gesture: Union[List[dict], np.ndarray], as_image: bool = True):
        """
        إما ترجع صورة (H,W,C) أو مصفوفة مفروضة (flatten).
        """
        img = self.gesture_to_image(gesture)
        if as_image:
            return img
        else:
            return img.flatten()

    def save_features_to_csv(self, gestures: List[dict], csv_path: str, include_header: bool = True, label_key: str = 'character'):
        """
        حفظ جميع الفيتشرات في CSV: كل صف = صورة مسطحة + label
        - gestures: قائمة gesture dict كما في parse_data()
        - csv_path: وجهة الحفظ
        """
        import csv as _csv
        rows = []
        for g in gestures:
            img = self.extract_features(g['points'], as_image=True)  # HxWxC
            flat = img.flatten()
            row = flat.tolist()
            row.insert(0, g.get('id', ''))
            row.insert(1, g.get(label_key, ''))
            rows.append(row)

        # header
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = _csv.writer(f)
            if include_header:
                header = ['gesture_id', 'character']
                num_pix = self.image_size * self.image_size * len(self.channels)
                header += [f'pix{i}' for i in range(num_pix)]
                writer.writerow(header)
            for r in rows:
                writer.writerow(r)
