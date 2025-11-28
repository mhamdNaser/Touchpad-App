# app/services/gesture_preprocessor.py
import numpy as np

class GesturePreprocessor:
    def __init__(self, normalize=True, resample_frames=50):
        self.normalize = normalize
        self.resample_frames = resample_frames  # طول ثابت لكل إيماءة بعد إعادة العينة

    def normalize_gesture(self, gesture):
        """
        تطبيع الإحداثيات لتكون بين 0 و 1 مهما كان حجم الشاشة
        gesture: قائمة dict لكل نقطة [{'x':..., 'y':..., 'pressure':...}, ...]
        """
        coords = np.array([[p['x'], p['y']] for p in gesture])
        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)
        coords_norm = (coords - min_vals) / (max_vals - min_vals + 1e-6)
        return coords_norm

    def resample_gesture(self, coords):
        """
        توحيد عدد الفريمات باستخدام interpolation
        coords: np.array [num_frames, 2]
        """
        num_frames = coords.shape[0]
        if num_frames == self.resample_frames:
            return coords
        # إنشاء مؤشرات جديدة للتسلسل الجديد
        idxs = np.linspace(0, num_frames-1, self.resample_frames)
        resampled = np.array([
            np.interp(idxs, np.arange(num_frames), coords[:, i])
            for i in range(coords.shape[1])
        ]).T
        return resampled

    def preprocess(self, gestures):
        """
        تطبيع وإعادة تشكيل جميع الإيماءات
        gestures: قائمة gesture dict
        خروج: np.array [num_gestures, resample_frames, 2]
        """
        processed = []
        for g in gestures:
            # تجاهل pressure لأن قيمته ثابتة
            coords = np.array([[p['x'], p['y']] for p in g['points']])
            if self.normalize:
                coords = self.normalize_gesture(g['points'])
            coords = self.resample_gesture(coords)
            processed.append(coords)
        return np.array(processed, dtype=np.float32)
