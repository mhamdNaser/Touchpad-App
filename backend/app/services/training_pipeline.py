# app/services/training_pipeline.py
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from app.services.gesture_data_loader import GestureDataLoader
from app.services.features_visualizer import FeatureEngineerVisualizer

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


class TrainingPipeline:
    def __init__(self, max_timesteps: int = 150):
        self.data_loader = GestureDataLoader(api_url="https://api.sydev.site/api/gestures")
        self.feature_engineer = FeatureEngineerVisualizer(max_timesteps=max_timesteps)

    # ======================================================
    # ✅ بناء نموذج 1D-CNN محسّن
    # ======================================================
    def build_cnn_model(self, input_shape, num_classes):
        model = Sequential([
            Conv1D(128, kernel_size=5, activation='relu', padding='same', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),

            Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),

            Conv1D(32, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            GlobalAveragePooling1D(),

            Dense(64, activation='relu'),
            Dropout(0.4),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.0005),
            metrics=['accuracy']
        )
        
        print(f"✅ Built CNN model with input shape {input_shape} and {num_classes} classes")
        return model

    # ======================================================
    # 🔧 تحويل مفاتيح dict إلى int
    # ======================================================
    def _convert_keys_to_int(self, d):
        result = {}
        for k, v in d.items():
            key = int(k) if isinstance(k, (np.integer, np.int64)) else k
            if isinstance(v, dict):
                result[key] = self._convert_keys_to_int(v)
            elif isinstance(v, (np.integer, np.int64)):
                result[key] = int(v)
            else:
                result[key] = v
        return result

    # ======================================================
    # 📊 رسم مصفوفة الالتباس
    # ======================================================
    def plot_confusion_matrix(self, y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    # ======================================================
    # 📈 رسم منحنى التدريب
    # ======================================================
    def plot_training_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # رسم دقة التدريب والتحقق
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # رسم فقدان التدريب والتحقق
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    # ======================================================
    # 🔍 فحص جودة البيانات قبل التدريب
    # ======================================================
    def _validate_data_quality(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """فحص جودة البيانات قبل البدء بالتدريب"""
        print("\n🔍 Validating data quality...")
        
        # فحص القيم غير الطبيعية
        def check_data_stats(data, name):
            flat_data = data.reshape(-1, data.shape[-1])
            print(f"📊 {name} - Shape: {data.shape}")
            print(f"   Range: [{data.min():.4f}, {data.max():.4f}]")
            print(f"   Mean: {data.mean():.4f}, Std: {data.std():.4f}")
            print(f"   NaN: {np.isnan(data).sum()}, Inf: {np.isinf(data).sum()}")
            
            # فحص إذا كانت البيانات كلها أصفار
            zero_ratio = (np.abs(flat_data) < 1e-6).mean()
            print(f"   Zero ratio: {zero_ratio:.4f}")
            
            return zero_ratio

        zero_ratios = []
        zero_ratios.append(check_data_stats(X_train, "X_train"))
        zero_ratios.append(check_data_stats(X_val, "X_val")) 
        zero_ratios.append(check_data_stats(X_test, "X_test"))

        # فحص توزيع التسميات
        print(f"🎯 Label distribution:")
        print(f"   Train: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"   Val: {dict(zip(*np.unique(y_val, return_counts=True)))}")
        print(f"   Test: {dict(zip(*np.unique(y_test, return_counts=True)))}")

        # تحذير إذا كانت نسبة الأصفار عالية
        if any(ratio > 0.8 for ratio in zero_ratios):
            print("⚠️  WARNING: High zero ratio detected - data might be over-padded")

        # تحذير إذا كانت الفئات غير متوازنة
        train_counts = np.bincount(y_train)
        if len(train_counts) > 0 and np.std(train_counts) > np.mean(train_counts) * 0.5:
            print("⚠️  WARNING: Class imbalance detected")

    # ======================================================
    # 📊 تصور توزيع الميزات قبل التدريب
    # ======================================================
    def visualize_feature_means(self, gestures_data):
        print("📊 Visualizing feature distributions before training...")
        self.feature_engineer.plot_feature_distribution(gestures_data)

    # ======================================================
    # 🚀 تنفيذ خط الأنابيب الكامل لتدريب 1D-CNN - مُحسّن
    # ======================================================
    def train_model(self):
        print("🚀 Starting 1D-CNN training pipeline...")

        # 1️⃣ تحميل البيانات
        gestures_data = self.data_loader.load_all_gestures()
        print(f"✅ Loaded {len(gestures_data)} gestures")
        if len(gestures_data) == 0:
            raise ValueError("❌ No gestures loaded from API.")

        # 1.5️⃣ تصور الميزات قبل التدريب
        self.visualize_feature_means(gestures_data)

        # 2️⃣ استخراج الميزات وتقسيم البيانات
        X_train, X_val, X_test, y_train, y_val, y_test, split_info, fixed_indices = self.feature_engineer.split_data(gestures_data)
        print("📋 Split information:")
        print(json.dumps(self._convert_keys_to_int(split_info), indent=2, ensure_ascii=False))

        # ✅ فحص جودة البيانات قبل الاستمرار
        self._validate_data_quality(X_train, X_val, X_test, y_train, y_val, y_test)

        # حفظ fixed_indices
        with open("split_indices.pkl", "wb") as f:
            pickle.dump(fixed_indices, f)

        # 3️⃣ ✅ تحسين تحويل التسميات إلى one-hot
        num_classes = len(self.feature_engineer.label_encoder.classes_)
        print(f"🎯 Number of classes: {num_classes}")
        print(f"🔠 Class names: {self.feature_engineer.label_encoder.classes_}")

        # فحص إذا كانت جميع التسميات موجودة في الـ encoder
        for y_set, name in [(y_train, 'train'), (y_val, 'val'), (y_test, 'test')]:
            unique_labels = np.unique(y_set)
            print(f"📝 {name} set unique labels: {unique_labels}")
            for label in unique_labels:
                if label not in self.feature_engineer.label_encoder.classes_:
                    print(f"⚠️  WARNING: Label {label} not in encoder classes!")

        # تحويل إلى one-hot encoding
        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        y_val_cat = to_categorical(y_val, num_classes=num_classes)
        y_test_cat = to_categorical(y_test, num_classes=num_classes)

        print(f"✅ One-hot shapes - Train: {y_train_cat.shape}, Val: {y_val_cat.shape}, Test: {y_test_cat.shape}")

        # 3.5️⃣ حساب أوزان الفئات
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weights = dict(enumerate(class_weights))
        print(f"⚖️ Class weights: {class_weights}")

        # 4️⃣ بناء النموذج
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self.build_cnn_model(input_shape, num_classes)
        
        # عرض ملخص النموذج
        model.summary()

        # 5️⃣ Callbacks محسنة
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=20,  # ✅ زيادة الصبر
            restore_best_weights=True,
            verbose=1
        )
        checkpoint = ModelCheckpoint(
            'arabic_gesture_cnn_best.h5', 
            monitor='val_accuracy', 
            save_best_only=True, 
            verbose=1
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=10,  # ✅ زيادة الصبر
            min_lr=1e-7,  # ✅ تقليل الحد الأدنى
            verbose=1
        )

        # 6️⃣ ✅ التدريب مع batch size محسّن
        batch_size = min(32, len(X_train) // 4)  # ✅ batch size ديناميكي
        if batch_size < 8:
            batch_size = 8
        print(f"🎯 Using batch size: {batch_size}")

        print("🏋️ Starting training...")
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=100,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint, reduce_lr],
            class_weight=class_weights,
            verbose=1
        )

        # 7️⃣ ✅ التقييم المحسّن
        print("📊 Evaluating model...")
        test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"✅ Test accuracy: {test_acc:.3f}")
        print(f"✅ Test loss: {test_loss:.3f}")

        # التنبؤات
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # ✅ فحص توزيع التنبؤات
        unique_pred, pred_counts = np.unique(y_pred, return_counts=True)
        print(f"📊 Prediction distribution: {dict(zip(unique_pred, pred_counts))}")

        print("\n📊 Classification report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.feature_engineer.label_encoder.classes_,
                                  zero_division=0))

        # رسم مصفوفة الالتباس
        self.plot_confusion_matrix(y_test, y_pred, self.feature_engineer.label_encoder.classes_)

        # رسم منحنى التدريب
        self.plot_training_history(history)

        # 8️⃣ ✅ حفظ النموذج والمكونات
        model.save("arabic_gesture_cnn_final.h5", save_format='h5')
        with open("scaler.pkl", "wb") as f:
            pickle.dump(self.feature_engineer.scaler, f)
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(self.feature_engineer.label_encoder, f)

        print("💾 Saved: model, scaler, and label encoder")

        # 9️⃣ ✅ حفظ بيانات الاختبار
        with open("X_test.pkl", "wb") as f:
            pickle.dump(X_test, f)
        with open("y_test.pkl", "wb") as f:
            pickle.dump(y_test, f)

        print("💾 Saved test data for future evaluation")

        return {
            'model': model,
            'history': history,
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'split_info': self._convert_keys_to_int(split_info),
            'predictions': {
                'y_true': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        }

    # ======================================================
    # 🔄 دالة مساعدة لإعادة التدريب مع معايرة محسنة
    # ======================================================
    def retrain_with_fixed_scaling(self, gestures_data=None):
        """إعادة التدريب مع إصلاح مشاكل المعايرة"""
        print("🔄 Retraining with fixed scaling...")
        
        if gestures_data is None:
            gestures_data = self.data_loader.load_all_gestures()
        
        # استخدام المعايرة المحسنة من FeatureEngineerVisualizer
        X_train, X_val, X_test, y_train, y_val, y_test, split_info, fixed_indices = self.feature_engineer.split_data(gestures_data)
        
        # متابعة التدريب الطبيعي
        return self.train_model()