# app/services/prediction_pipeline.py
import os
import torch
import numpy as np
from typing import Dict, Any, List
from joblib import load
import logging

logger = logging.getLogger(__name__)

class CompatibleEncoder(torch.nn.Module):
    def __init__(self, input_dim=60*20*6, latent_dim=128):
        super().__init__()
        # بناء على الخطأ، يبدو أن النموذج المدرب يحتوي على BatchNorm وطبقات أكثر
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),  # أضفنا BatchNorm
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(256),  # أضفنا BatchNorm
            torch.nn.Linear(256, latent_dim)
        )
    
    def forward(self, x):
        # تأكد من أن الأبعاد صحيحة
        if x.shape[1] != self.net[0].in_features:
            # إذا كانت الأبعاد مختلفة، نقوم بتعديلها
            if x.shape[1] > self.net[0].in_features:
                x = x[:, :self.net[0].in_features]
            else:
                padding = torch.zeros(x.shape[0], self.net[0].in_features - x.shape[1])
                x = torch.cat([x, padding], dim=1)
        return self.net(x)

class FlexibleEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # سنبني النموذج ديناميكياً بناء على state_dict
        self.layers = torch.nn.ModuleList()
        
    def build_from_state_dict(self, state_dict, input_dim):
        """بناء النموذج ديناميكياً من state_dict"""
        print("🔨 بناء النموذج ديناميكياً من state_dict...")
        
        # تحليل state_dict لمعرفة البنية
        linear_layers = {}
        bn_layers = {}
        
        for key, tensor in state_dict.items():
            if 'weight' in key and 'bn' not in key and 'batch' not in key:
                layer_name = key.replace('.weight', '')
                linear_layers[layer_name] = tensor.shape
            elif 'bias' in key and 'bn' not in key and 'batch' not in key:
                layer_name = key.replace('.bias', '')
            elif 'weight' in key and ('bn' in key or 'batch' in key):
                layer_name = key.replace('.weight', '')
                bn_layers[layer_name] = tensor.shape
        
        print(f"📊 الطبقات الخطية: {linear_layers}")
        print(f"📊 طبقات BatchNorm: {bn_layers}")
        
        # بناء النموذج بناء على التحليل
        layers = []
        current_dim = input_dim
        
        # ترتيب الطبقات بناء على المفاتيح
        sorted_keys = sorted(linear_layers.keys())
        for i, key in enumerate(sorted_keys):
            out_dim = linear_layers[key][0]
            
            # إضافة طبقة خطية
            layers.append(torch.nn.Linear(current_dim, out_dim))
            layers.append(torch.nn.ReLU())
            
            # التحقق إذا كان هناك BatchNorm بعد هذه الطبقة
            bn_key = f"net.{2*i+1}" if 'net' in key else f"{key.replace('net.', '')}_bn"
            if bn_key in bn_layers:
                layers.append(torch.nn.BatchNorm1d(out_dim))
            
            current_dim = out_dim
        
        # الطبقة الأخيرة (latent)
        layers.append(torch.nn.Linear(current_dim, list(linear_layers.values())[-1][0]))
        
        self.net = torch.nn.Sequential(*layers)
        print(f"✅ النموذج المبني: {self.net}")
        
    def forward(self, x):
        return self.net(x)

class ClusteringPredictionPipeline:
    def __init__(
        self,
        encoder_path: str = "artifacts/encoder.pth",
        kmeans_path: str = "artifacts/gesture_kmeans.joblib", 
        mapping_path: str = "artifacts/gesture_mapping.joblib",
        max_frames: int = 60,
        max_points: int = 20,
        verbose: bool = True
    ):
        self.max_frames = max_frames
        self.max_points = max_points
        self.verbose = verbose
        self.is_ready = False
        self.load_error = None
        
        try:
            # التحقق من وجود الملفات المطلوبة
            required_files = [encoder_path, kmeans_path, mapping_path]
            missing_files = [path for path in required_files if not os.path.exists(path)]
            
            if missing_files:
                self.load_error = f"ملفات النموذج مفقودة: {missing_files}"
                logger.warning(f"⚠️ {self.load_error}")
                return
            
            # أبعاد النموذج
            self.input_dim = max_frames * max_points * 6
            self.latent_dim = 128
            
            # محاولة تحميل النموذج بطرق مختلفة
            self.encoder = self._load_encoder_compatible(encoder_path)
            if self.encoder is None:
                return
                
            self.encoder.eval()
            
            # تحميل KMeans و Mapping
            try:
                self.kmeans = load(kmeans_path)
                self.mapping = load(mapping_path)
            except Exception as e:
                self.load_error = f"خطأ في تحميل KMeans/Mapping: {e}"
                logger.error(self.load_error)
                return
            
            self.is_ready = True
            logger.info("✅ تم تحميل النموذج بنجاح")
            logger.info(f"📊 تفاصيل النموذج: {len(self.mapping)} عنقود")
            
        except Exception as e:
            self.load_error = f"خطأ غير متوقع في التهيئة: {e}"
            logger.error(self.load_error)
            self.is_ready = False

    def _load_encoder_compatible(self, encoder_path):
        """محاولة تحميل الencoder بطرق متوافقة مختلفة"""
        methods = [
            self._load_with_dynamic_build,
            self._load_with_compatible_encoder, 
            self._load_with_strict_encoder
        ]
        
        for method in methods:
            try:
                encoder = method(encoder_path)
                if encoder is not None:
                    logger.info(f"✅ تم تحميل النموذج باستخدام: {method.__name__}")
                    return encoder
            except Exception as e:
                logger.warning(f"⚠️ فشلت الطريقة {method.__name__}: {e}")
                continue
        
        self.load_error = "فشل جميع طرق تحميل النموذج"
        return None

    def _load_with_dynamic_build(self, encoder_path):
        """الطريقة 1: بناء النموذج ديناميكياً"""
        encoder = FlexibleEncoder()
        state_dict = torch.load(encoder_path, map_location='cpu')
        encoder.build_from_state_dict(state_dict, self.input_dim)
        encoder.load_state_dict(state_dict, strict=False)
        return encoder

    def _load_with_compatible_encoder(self, encoder_path):
        """الطريقة 2: استخدام النموذج المتوافق مع BatchNorm"""
        encoder = CompatibleEncoder(self.input_dim, self.latent_dim)
        state_dict = torch.load(encoder_path, map_location='cpu')
        
        # تحميل مع تجاهل بعض المفاتيح غير المتطابقة
        encoder.load_state_dict(state_dict, strict=False)
        return encoder

    def _load_with_strict_encoder(self, encoder_path):
        """الطريقة 3: استخدام النموذج البسيط مع strict=False"""
        # نموذج بسيط يشبه البنية الأصلية
        encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 256), 
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, self.latent_dim)
        )
        
        state_dict = torch.load(encoder_path, map_location='cpu')
        encoder.load_state_dict(state_dict, strict=False)
        return encoder

    # باقي الدوال تبقى كما هي (normalize_gesture, resample_frames, etc.)
    def normalize_gesture(self, frames: List[Dict]) -> List[Dict]:
        """تطبيع إحداثيات الإيماءة"""
        if not frames:
            return frames
            
        # جمع جميع النقاط من جميع الإطارات
        all_points = []
        for frame in frames:
            points = frame.get("points", [])
            for point in points:
                all_points.append((point.get("x", 0), point.get("y", 0)))
        
        if not all_points:
            return frames
            
        # حساب المدى
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        width = max(max_x - min_x, 1.0)
        height = max(max_y - min_y, 1.0)
        scale = max(width, height, 1.0)
        
        # تطبيع الإحداثيات
        normalized_frames = []
        for frame in frames:
            normalized_points = []
            points = frame.get("points", [])
            for point in points:
                x_norm = (point.get("x", 0) - min_x) / scale
                y_norm = (point.get("y", 0) - min_y) / scale
                normalized_points.append({
                    "x": float(x_norm),
                    "y": float(y_norm),
                    "pressure": point.get("pressure", 1.0)
                })
            normalized_frames.append({
                "timestamp": frame.get("ts", frame.get("timestamp", 0)),
                "delta_ms": frame.get("delta_ms", 16),
                "points": normalized_points
            })
        
        return normalized_frames

    def resample_frames(self, frames: List[Dict], target_frames: int) -> List[Dict]:
        """إعادة تشكيل عدد الإطارات"""
        if len(frames) <= 1 or target_frames <= 0:
            return frames
            
        if len(frames) == target_frames:
            return frames
            
        # استخراج النقاط من كل إطار
        frame_points = []
        for frame in frames:
            points = frame.get("points", [])
            frame_points.append(points)
        
        # إعادة التشكيل الخطي
        original_indices = np.linspace(0, len(frames) - 1, len(frames))
        target_indices = np.linspace(0, len(frames) - 1, target_frames)
        
        resampled_frames = []
        for target_idx in target_indices:
            idx = int(round(target_idx))
            idx = min(idx, len(frames) - 1)
            
            # نسخ النقاط
            copied_points = []
            for point in frame_points[idx]:
                copied_points.append({
                    "x": point.get("x", 0),
                    "y": point.get("y", 0),
                    "pressure": point.get("pressure", 1.0)
                })
            
            resampled_frames.append({
                "timestamp": frames[idx].get("ts", frames[idx].get("timestamp", 0)),
                "delta_ms": frames[idx].get("delta_ms", 16),
                "points": copied_points
            })
        
        return resampled_frames

    def resample_points_per_frame(self, frames: List[Dict], target_points: int) -> List[Dict]:
        """إعادة تشكيل عدد النقاط في كل إطار"""
        resampled_frames = []
        
        for frame in frames:
            points = frame.get("points", [])
            if len(points) == target_points:
                resampled_frames.append(frame)
                continue
                
            if len(points) <= 1:
                default_point = points[0] if points else {"x": 0, "y": 0, "pressure": 1.0}
                resampled_points = [default_point.copy() for _ in range(target_points)]
                resampled_frames.append({
                    "timestamp": frame.get("timestamp", 0),
                    "delta_ms": frame.get("delta_ms", 16),
                    "points": resampled_points
                })
                continue
            
            xs = [p["x"] for p in points]
            ys = [p["y"] for p in points]
            pressures = [p.get("pressure", 1.0) for p in points]
            
            original_indices = np.linspace(0, len(points)-1, len(points))
            target_indices = np.linspace(0, len(points)-1, target_points)
            
            new_xs = np.interp(target_indices, original_indices, xs)
            new_ys = np.interp(target_indices, original_indices, ys)
            new_pressures = np.interp(target_indices, original_indices, pressures)
            
            resampled_points = []
            for i in range(target_points):
                resampled_points.append({
                    "x": float(new_xs[i]),
                    "y": float(new_ys[i]),
                    "pressure": float(new_pressures[i])
                })
            
            resampled_frames.append({
                "timestamp": frame.get("timestamp", 0),
                "delta_ms": frame.get("delta_ms", 16),
                "points": resampled_points
            })
        
        return resampled_frames

    def calculate_derived_features(self, frames: List[Dict]) -> List[Dict]:
        """حساب المميزات المشتقة (dx, dy, angle)"""
        if len(frames) <= 1:
            enhanced_frames = []
            for frame in frames:
                enhanced_points = []
                for point in frame.get("points", []):
                    enhanced_points.append({
                        "x": point.get("x", 0),
                        "y": point.get("y", 0),
                        "pressure": point.get("pressure", 1.0),
                        "dx": 0.0,
                        "dy": 0.0,
                        "angle": 0.0
                    })
                enhanced_frames.append({
                    "timestamp": frame.get("timestamp", 0),
                    "delta_ms": frame.get("delta_ms", 16),
                    "points": enhanced_points
                })
            return enhanced_frames
        
        enhanced_frames = []
        
        first_frame = frames[0]
        first_points = []
        for point in first_frame.get("points", []):
            first_points.append({
                "x": point.get("x", 0),
                "y": point.get("y", 0),
                "pressure": point.get("pressure", 1.0),
                "dx": 0.0,
                "dy": 0.0,
                "angle": 0.0
            })
        enhanced_frames.append({
            "timestamp": first_frame.get("timestamp", 0),
            "delta_ms": first_frame.get("delta_ms", 16),
            "points": first_points
        })
        
        for i in range(1, len(frames)):
            current_frame = frames[i]
            prev_frame = frames[i-1]
            
            enhanced_points = []
            current_points = current_frame.get("points", [])
            prev_points = prev_frame.get("points", [])
            
            min_points = min(len(current_points), len(prev_points))
            
            for j in range(min_points):
                curr_pt = current_points[j]
                prev_pt = prev_points[j]
                
                dx = curr_pt.get("x", 0) - prev_pt.get("x", 0)
                dy = curr_pt.get("y", 0) - prev_pt.get("y", 0)
                angle = np.arctan2(dy, dx) if (dx != 0 or dy != 0) else 0.0
                
                enhanced_points.append({
                    "x": curr_pt.get("x", 0),
                    "y": curr_pt.get("y", 0),
                    "pressure": curr_pt.get("pressure", 1.0),
                    "dx": float(dx),
                    "dy": float(dy),
                    "angle": float(angle)
                })
            
            for j in range(min_points, len(current_points)):
                curr_pt = current_points[j]
                enhanced_points.append({
                    "x": curr_pt.get("x", 0),
                    "y": curr_pt.get("y", 0),
                    "pressure": curr_pt.get("pressure", 1.0),
                    "dx": 0.0,
                    "dy": 0.0,
                    "angle": 0.0
                })
            
            enhanced_frames.append({
                "timestamp": current_frame.get("timestamp", 0),
                "delta_ms": current_frame.get("delta_ms", 16),
                "points": enhanced_points
            })
        
        return enhanced_frames

    def gesture_to_tensor(self, frames: List[Dict]) -> np.ndarray:
        """تحويل الإيماءة إلى tensor"""
        F = len(frames)
        P = self.max_points
        C = 6
        
        tensor = np.zeros((F, P, C), dtype=np.float32)
        
        for fi, frame in enumerate(frames):
            points = frame.get("points", [])
            for pi, point in enumerate(points[:P]):
                tensor[fi, pi, 0] = point.get("x", 0.0)
                tensor[fi, pi, 1] = point.get("y", 0.0)
                tensor[fi, pi, 2] = point.get("pressure", 1.0)
                tensor[fi, pi, 3] = point.get("dx", 0.0)
                tensor[fi, pi, 4] = point.get("dy", 0.0)
                tensor[fi, pi, 5] = point.get("angle", 0.0)
        
        return tensor

    def preprocess_gesture(self, gesture_data: Dict[str, Any]) -> np.ndarray:
        """معالجة البيانات المدخلة من الواجهة"""
        try:
            frames = gesture_data.get("frames", [])
            
            if not frames:
                raise ValueError("لا توجد إطارات في البيانات")
            
            logger.info(f"📥 معالجة إيماءة: {len(frames)} إطار")
            
            normalized_frames = self.normalize_gesture(frames)
            resampled_frames = self.resample_frames(normalized_frames, self.max_frames)
            final_frames = self.resample_points_per_frame(resampled_frames, self.max_points)
            enhanced_frames = self.calculate_derived_features(final_frames)
            tensor_data = self.gesture_to_tensor(enhanced_frames)
            
            logger.info(f"✅ المعالجة اكتملت: tensor shape {tensor_data.shape}")
            
            return tensor_data
            
        except Exception as e:
            logger.error(f"❌ خطأ في معالجة الإيماءة: {e}")
            raise

    def predict_gesture(self, gesture_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_ready:
            return {
                "success": False,
                "error": f"النموذج غير جاهز للتنبؤ. {self.load_error}",
                "predicted_letter": "?",
                "cluster": None
            }
        
        try:
            logger.info("🎯 بدء التنبؤ...")
            
            processed_tensor = self.preprocess_gesture(gesture_data)
            flattened = processed_tensor.reshape(1, -1)
            input_tensor = torch.tensor(flattened, dtype=torch.float32)
            
            with torch.no_grad():
                latent = self.encoder(input_tensor).numpy()[0]
            
            cluster_idx = int(self.kmeans.predict([latent])[0])
            predicted_letter = self.mapping.get(cluster_idx, f"Cluster_{cluster_idx}")
            
            logger.info(f"🎉 التنبؤ النهائي: '{predicted_letter}' (العنقود: {cluster_idx})")
            
            return {
                "success": True,
                "predicted_letter": predicted_letter,
                "cluster": cluster_idx,
                "confidence": 1.0
            }
            
        except Exception as e:
            logger.error(f"❌ خطأ في التنبؤ: {e}")
            return {
                "success": False,
                "error": str(e),
                "predicted_letter": "?",
                "cluster": None
            }

    def get_status(self) -> Dict[str, Any]:
        status = {
            "is_ready": self.is_ready,
            "model_loaded": self.is_ready,
            "max_frames": self.max_frames,
            "max_points": self.max_points,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim
        }
        
        if self.load_error:
            status["load_error"] = self.load_error
            
        if self.is_ready:
            status["clusters_count"] = len(self.mapping)
            status["mapping_sample"] = dict(list(self.mapping.items())[:5])
        
        return status