# app/services/gesture_data_loader.py (UPDATED VERSION)
from typing import List, Dict, Optional, Tuple
import numpy as np
import requests

class GestureDataLoader:
    def __init__(self, api_url: str = "https://api.sydev.site/api/gestures", per_page: int = 50,
                 target_frames: int = 60, target_points: int = 20):
        self.api_url = api_url
        self.per_page = per_page
        self.target_frames = target_frames
        self.target_points = target_points
        self.session = requests.Session()
        self.session.timeout = 30
        
    def _extract_points_from_gesture(self, gesture: Dict) -> List[Dict]:
        """استخراج النقاط من الإيماءة (بطرق مختلفة)"""
        points = []
        
        # الطريقة 1: النقاط مباشرة في الجيشر
        if 'points' in gesture and gesture['points']:
            return gesture['points']
        
        # الطريقة 2: النقاط داخل frames
        if 'frames' in gesture and gesture['frames']:
            for frame in gesture['frames']:
                if 'points' in frame and frame['points']:
                    points.extend(frame['points'])
            return points
        
        # الطريقة 3: البيانات الخام
        if 'raw_data' in gesture and gesture['raw_data']:
            # هنا تحتاج لمعالجة raw_data حسب الهيكل
            pass
        
        return points

    def _create_frames_from_points(self, points: List[Dict]) -> List[Dict]:
        """إنشاء إطارات من النقاط المباشرة"""
        if not points:
            return []
            
        # تجميع النقاط حسب frame_id إذا موجود
        frames_dict = {}
        for point in points:
            frame_id = point.get('frame_id', 0)
            if frame_id not in frames_dict:
                frames_dict[frame_id] = {
                    'timestamp': point.get('timestamp', 0),
                    'delta_ms': point.get('delta_ms', 16),
                    'points': []
                }
            
            frames_dict[frame_id]['points'].append({
                'x': point.get('x', 0.0),
                'y': point.get('y', 0.0),
                'pressure': point.get('pressure', 1.0)
            })
        
        return list(frames_dict.values())

    def _normalize_gesture(self, frames: List[Dict]) -> List[Dict]:
        """تطبيع الإيماءة"""
        if not frames:
            return frames
            
        # جمع جميع النقاط
        all_points = []
        for frame in frames:
            all_points.extend(frame['points'])
        
        if not all_points:
            return frames
            
        # حساب الصندوق المحيط
        xs = [p['x'] for p in all_points]
        ys = [p['y'] for p in all_points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        width = max_x - min_x
        height = max_y - min_y
        
        if width == 0: width = 1
        if height == 0: height = 1
        
        # التطبيع
        scale = 2.0 / max(width, height)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        normalized_frames = []
        for frame in frames:
            normalized_points = []
            for point in frame['points']:
                normalized_points.append({
                    'x': (point['x'] - center_x) * scale,
                    'y': (point['y'] - center_y) * scale,
                    'pressure': point.get('pressure', 1.0)
                })
            
            normalized_frames.append({
                'timestamp': frame.get('timestamp', 0),
                'delta_ms': frame.get('delta_ms', 16),
                'points': normalized_points
            })
        
        return normalized_frames

    def _resample_frames(self, frames: List[Dict], target_frames: int) -> List[Dict]:
        """إعادة العينة الزمنية"""
        if len(frames) <= 1 or len(frames) == target_frames:
            return frames
            
        # حساب الأوقات
        timestamps = [f.get('timestamp', i * 16) for i, f in enumerate(frames)]
        total_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1
        target_times = np.linspace(timestamps[0], timestamps[-1], target_frames)
        
        resampled_frames = []
        for target_time in target_times:
            # البحث عن الإطار المناسب
            idx = min(len(frames)-1, max(0, int((target_time - timestamps[0]) / total_time * (len(frames)-1))))
            resampled_frames.append(frames[idx])
        
        return resampled_frames

    def _process_gesture(self, gesture: Dict) -> Optional[Dict]:
        """معالجة إيماءة واحدة"""
        # استخراج النقاط
        raw_points = self._extract_points_from_gesture(gesture)
        
        if not raw_points:
            return None
        
        # إنشاء إطارات من النقاط
        frames = self._create_frames_from_points(raw_points)
        
        if not frames:
            return None
        
        # التطبيع
        frames = self._normalize_gesture(frames)
        
        # إعادة العينة الزمنية
        frames = self._resample_frames(frames, self.target_frames)
        
        # إعادة عينة النقاط لكل إطار
        for i, frame in enumerate(frames):
            points = frame['points']
            if len(points) != self.target_points:
                # إعادة عينة بسيطة
                if len(points) > self.target_points:
                    # تقليل
                    indices = np.linspace(0, len(points)-1, self.target_points).astype(int)
                    frame['points'] = [points[i] for i in indices]
                else:
                    # تكرار آخر نقطة
                    last_point = points[-1] if points else {'x': 0, 'y': 0, 'pressure': 1.0}
                    while len(frame['points']) < self.target_points:
                        frame['points'].append(last_point.copy())
        
        return {
            'gesture_id': gesture.get('id'),
            'character': gesture.get('character'),
            'frames': frames,
            'original_points': len(raw_points),
            'processed_frames': len(frames)
        }

    def load_all_gestures(self) -> List[Dict]:
        """تحميل جميع الإيماءات من كل الصفحات"""
        page = 1
        all_raw_gestures = []
        total_gestures = 0
        
        print("📥 جاري تحميل البيانات من API...")
        while True:
            url = f"{self.api_url}?page={page}&per_page={self.per_page}"
            try:
                response = self.session.get(url)
                response.raise_for_status()
                data = response.json()
                gestures = data.get("data", [])
                
                if not gestures:
                    break
                    
                all_raw_gestures.extend(gestures)
                total_gestures += len(gestures)
                page += 1
                    
            except Exception as e:
                print(f"❌ خطأ في تحميل الصفحة {page}: {e}")
                break
        
        print(f"📊 إجمالي الإيماءات المجمعة: {total_gestures}")
        
        # معالجة الإيماءات
        processed_gestures = []
        successful = 0
        
        for raw_gesture in all_raw_gestures:
            try:
                processed = self._process_gesture(raw_gesture)
                if processed:
                    processed_gestures.append(processed)
                    successful += 1
            except Exception:
                continue
        
        print(f"🎉 تم معالجة {successful}/{total_gestures} إيماءة بنجاح")
        
        # تحليل توزيع الحروف
        char_distribution = {}
        for gesture in processed_gestures:
            char = gesture.get('character', 'unknown')
            char_distribution[char] = char_distribution.get(char, 0) + 1
        
        print("📈 توزيع الحروف:")
        for char, count in char_distribution.items():
            print(f"   - {char}: {count} إيماءة")
        
        return processed_gestures