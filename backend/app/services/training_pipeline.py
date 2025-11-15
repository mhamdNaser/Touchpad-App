# app/services/training_pipeline.py
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from app.services.gesture_data_loader import GestureDataLoader
from app.services.features_visualizer import ProductionFeatureExtractor


class TrainingPipeline:
    def __init__(self, max_timesteps: int = 50, verbose: bool = True):
        self.data_loader = GestureDataLoader()
        self.feature_extractor = ProductionFeatureExtractor(max_timesteps=max_timesteps, verbose=verbose)
        self.max_timesteps = max_timesteps
        self.verbose = verbose
        self.label_encoder = LabelEncoder()

    # ======================================================
    # ✅ بناء نموذج 1D-CNN محسّن
    # ======================================================
    def build_cnn_model(self, input_shape, num_classes):
        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.5),

            Conv1D(16, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            GlobalAveragePooling1D(),

            Dense(32, activation='relu'),
            Dropout(0.6),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.0005),
            metrics=['accuracy']
        )
        
        if self.verbose:
            print(f"✅ Built CNN model with input shape {input_shape} and {num_classes} classes")
        return model

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
        print("\n🔍 Validating data quality...")

        def check_data_stats(data, name):
            flat_data = data.reshape(-1, data.shape[-1])
            print(f"📊 {name} - Shape: {data.shape}")
            print(f"   Range: [{data.min():.4f}, {data.max():.4f}]")
            print(f"   Mean: {data.mean():.4f}, Std: {data.std():.4f}")
            print(f"   NaN: {np.isnan(data).sum()}, Inf: {np.isinf(data).sum()}")
            zero_ratio = (np.abs(flat_data) < 1e-6).mean()
            print(f"   Zero ratio: {zero_ratio:.4f}")
            return zero_ratio

        zero_ratios = [
            check_data_stats(X_train, "X_train"),
            check_data_stats(X_val, "X_val"),
            check_data_stats(X_test, "X_test")
        ]

        print(f"🎯 Label distribution:")
        for name, y_set in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            unique, counts = np.unique(y_set, return_counts=True)
            print(f"   {name}: {dict(zip(unique, counts))}")

        if any(ratio > 0.8 for ratio in zero_ratios):
            print("⚠️  WARNING: High zero ratio detected - data might be over-padded")

    # ======================================================
    # 🚀 تنفيذ خط الأنابيب الكامل لتدريب 1D-CNN
    # ======================================================
    def train_model(self):
        print("🚀 Starting 1D-CNN training pipeline...")

        # 1️⃣ تحميل البيانات
        gestures_data = self.data_loader.load_all_gestures()
        print(f"✅ Loaded {len(gestures_data)} gestures")
        if len(gestures_data) == 0:
            raise ValueError("❌ No gestures loaded from API.")

        # 2️⃣ استخراج الميزات sequences per-frame
        X = []
        y = []
        for g in gestures_data:
            seq = self.feature_extractor._gesture_to_sequence(g)
            X.append(seq)
            y.append(g['character'])

        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        # 3️⃣ تحويل الحروف إلى أرقام
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(self.label_encoder.classes_)
        print(f"🎯 Number of classes: {num_classes}")
        print(f"🔠 Class names: {self.label_encoder.classes_}")

        # 4️⃣ تقسيم البيانات
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        # 5️⃣ تحويل إلى one-hot
        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        y_val_cat = to_categorical(y_val, num_classes=num_classes)
        y_test_cat = to_categorical(y_test, num_classes=num_classes)

        # 6️⃣ حساب أوزان الفئات
        try:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weights = dict(enumerate(class_weights))
            print(f"⚖️ Class weights: {class_weights}")
        except Exception as e:
            print(f"⚠️ Could not compute class weights: {e}")
            class_weights = None

        # 7️⃣ بناء النموذج
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self.build_cnn_model(input_shape, num_classes)
        model.summary()

        # 8️⃣ Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
        checkpoint = ModelCheckpoint('arabic_gesture_cnn_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)

        # 9️⃣ التدريب
        batch_size = min(32, len(X_train) // 4)
        batch_size = max(batch_size, 8)
        print(f"🎯 Using batch size: {batch_size}")

        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=100,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint, reduce_lr],
            class_weight=class_weights,
            verbose=1
        )

        # 10️⃣ التقييم
        test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"✅ Test accuracy: {test_acc:.3f}")
        print(f"✅ Test loss: {test_loss:.3f}")

        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        print("\n📊 Classification report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_, zero_division=0))
        self.plot_confusion_matrix(y_test, y_pred, self.label_encoder.classes_)
        self.plot_training_history(history)

        # 11️⃣ حفظ النموذج والمكونات
        model.save("arabic_gesture_cnn_final.h5", save_format='h5')
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        with open("X_test.pkl", "wb") as f:
            pickle.dump(X_test, f)
        with open("y_test.pkl", "wb") as f:
            pickle.dump(y_test, f)
        print("💾 Saved model, label encoder, and test data")

        return {
            'model': model,
            'history': history,
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'predictions': {
                'y_true': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        }