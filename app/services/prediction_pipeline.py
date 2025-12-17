import numpy as np
from keras.models import load_model, Model
from typing import cast
from app.services.feature_extractor import GestureFeatureExtractor
import os


class PredictionPipeline:
    def __init__(self, model_path=None, label_path=None, verbose=False):
        self.verbose = verbose

        # مسارات افتراضية
        model_dir = "ai_model"
        if model_path is None:
            model_path = os.path.join(model_dir, "best_model.h5")
        if label_path is None:
            label_path = os.path.join(model_dir, "label_classes.npy")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label classes file not found: {label_path}")

        if self.verbose:
            print(f"📂 Loading model: {model_path}")
            print(f"📂 Loading labels: {label_path}")

        # تحميل النموذج
        self.model = cast(Model, load_model(model_path))
        if self.model is None:
            raise RuntimeError(f"Failed to load model from: {model_path}")

        # تحميل أسماء الأصناف
        self.class_labels = np.load(label_path, allow_pickle=True).tolist()

        # مستخرج الميزات
        self.extractor = GestureFeatureExtractor(
            image_size=64,
            thickness=1.5,
            channels=("stroke", "velocity")
        )

        if self.verbose:
            print(f"✅ Model loaded. Classes: {len(self.class_labels)}")

    # -----------------------------------------------------
    def gesture_to_image(self, gesture_dict):
        frames = gesture_dict.get("frames", [])
        points = []

        # جمع النقاط
        for frame in frames:
            points.extend(frame.get("points", []))

        if not points:
            raise ValueError("No valid points in gesture.")

        # استخراج صورة = (64,64,2)
        img = self.extractor.extract_features(points, as_image=True)

        # توحيد المدخلات إلى قناة واحدة فقط (64,64,1)
        if img.ndim == 3 and img.shape[-1] == 2:
            img = img[..., :1]

        # إضافة batch dimension
        return np.expand_dims(img, axis=0)

    # -----------------------------------------------------
    def predict_gesture_top3(self, gesture_dict):
        img = self.gesture_to_image(gesture_dict)

        # تنبؤ
        pred_probs = self.model.predict(img)[0]

        # أفضل 3
        top3_idx = np.argsort(pred_probs)[::-1][:3]
        top3_chars = [self.class_labels[i] for i in top3_idx]
        top3_conf = [float(pred_probs[i]) for i in top3_idx]

        return {
            "predicted_char": top3_chars[0],
            "confidence": top3_conf[0],
            "top3": [
                {"char": c, "confidence": p}
                for c, p in zip(top3_chars, top3_conf)
            ],
        }

    # -----------------------------------------------------
    def get_model_info(self):
        return {
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "num_classes": len(self.class_labels),
        }
