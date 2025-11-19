import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from app.services.gesture_data_loader import GestureDataLoader
from app.services.advanced_feature_extractor import AdvancedFeatureExtractor


class TrainingPipeline:
    def __init__(self, verbose:bool=True):
        self.data_loader = GestureDataLoader()
        self.feature_extractor = AdvancedFeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.verbose = verbose

        self.output_dir = "ai_model"
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_training_history(self, history):
        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_history.png"))
        plt.show()
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.show()
        plt.close()
        print("💾 Saved confusion matrix in ai_model/confusion_matrix.png")

    def build_model(self, input_shape, num_classes):
        """بناء نموذج لمعالجة الميزات العالمية (2D)"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(0.001),
            metrics=['accuracy']
        )

        return model

    def train_model(self):
        print("🚀 Starting training pipeline with global features...")
        gestures = self.data_loader.load_all_gestures()

        if len(gestures) == 0:
            raise ValueError("❌ No gestures loaded")

        # استخراج الميزات العالمية
        X, y = [], []
        for g in gestures:
            try:
                features = self.feature_extractor.gesture_to_feature_vector(g)
                X.append(features)
                y.append(g["character"])
            except Exception as e:
                print(f"⚠️ Skipping gesture {g.get('gesture_id')}: {e}")
                continue

        if len(X) == 0:
            raise ValueError("❌ No features extracted")

        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        print(f"📊 Data shape: {X.shape}")
        print(f"🎯 Classes: {np.unique(y)}")
        print(f"📈 Features: {self.feature_extractor.get_feature_names()}")

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(self.label_encoder.classes_)

        # تطبيع الميزات
        X_scaled = self.scaler.fit_transform(X)

        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )

        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        y_test_cat = to_categorical(y_test, num_classes=num_classes)

        # حساب الأوزان
        try:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weights = dict(enumerate(class_weights))
            print(f"⚖️ Class weights: {class_weights}")
        except Exception as e:
            print(f"⚠️ Could not compute class weights: {e}")
            class_weights = None

        # بناء النموذج
        input_shape = X_train.shape[1]  # عدد الميزات
        model = self.build_model(input_shape, num_classes)

        print("🧮 Model Summary:")
        model.summary()

        batch_size = 32
        if len(X_train) < 32:
            batch_size = 8

        # callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1)
        checkpoint = ModelCheckpoint(
            os.path.join(self.output_dir, 'best_model.keras'),
            monitor='val_accuracy', save_best_only=True, verbose=1
        )
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)

        # التدريب
        print("🎯 Starting training...")
        history = model.fit(
            X_train, y_train_cat,
            validation_split=0.2,
            epochs=100,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint, reduce_lr],
            class_weight=class_weights,
            verbose=1
        )

        # الرسوم البيانية
        self.plot_training_history(history)

        # التقييم
        print("📊 Evaluating model...")
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_, zero_division=0))

        self.plot_confusion_matrix(y_test, y_pred)

        # حفظ النموذج
        model.save(os.path.join(self.output_dir, "final_model.keras"))
        with open(os.path.join(self.output_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(self.label_encoder, f)
        with open(os.path.join(self.output_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        # حفظ معلومات الميزات
        feature_info = {
            'feature_names': self.feature_extractor.get_feature_names(),
            'feature_dimension': self.feature_extractor.get_feature_dimension()
        }
        with open(os.path.join(self.output_dir, "feature_info.pkl"), "wb") as f:
            pickle.dump(feature_info, f)

        test_accuracy = np.mean(y_pred == y_test)
        print(f"🎉 Final Test Accuracy: {test_accuracy:.4f}")

        return {
            "test_accuracy": float(test_accuracy),
            "model": model,
            "history": history,
            "feature_names": self.feature_extractor.get_feature_names()
        }