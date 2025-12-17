# app/services/training_pipeline.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from kagglehub import dataset_download

import os
import cv2
import numpy as np

def load_external_images(folder, target_size=(64, 64)):
    ARABIC_LABELS = [
        "ا","ب","ت","ث","ج","ح","خ","د","ذ","ر","ز","س","ش","ص","ض",
        "ط","ظ","ع","غ","ف","ق","ك","ل","م","ن","ه","و","ي"
    ]
    
    X, y = [], []

    if not os.path.exists(folder):
        return np.array([]), np.array([])

    for class_folder in sorted(os.listdir(folder)):
        class_path = os.path.join(folder, class_folder)
        if not os.path.isdir(class_path):
            continue

        try:
            class_idx = int(class_folder) - 1  # تصحيح الفهرس
            if class_idx < 0 or class_idx >= len(ARABIC_LABELS):
                print(f"Skipping invalid class index: {class_idx+1}")
                continue
        except ValueError:
            print(f"Skipping non-integer folder name: {class_folder}")
            continue

        for file_name in os.listdir(class_path):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(class_path, file_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Failed to read image: {img_path}")
                    continue

                img = cv2.resize(img, target_size)
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, -1)

                X.append(img)
                y.append(ARABIC_LABELS[class_idx])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=object)
    
    print(f"Loaded {len(X)} images from {folder}")
    return X, y

class GestureTrainer:
    def __init__(self, image_size=(64, 64), model_dir="ai_model"):
        self.image_size = image_size
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.le = LabelEncoder()
        self.model = None
        self.num_classes = None

    # ----------------------------------------------------------
    #             دمج صور الإيماءات + صور كاغل
    # ----------------------------------------------------------
    def prepare_dataset(self, X_gestures, y_gestures):
        print("\n🌐 Downloading external dataset from Kaggle ...")
        kaggle_path = dataset_download("mohdrehanhaider/arabic-characters")

        train_folder = os.path.join(kaggle_path, "Arabic Character Dataset", "Train Arabic")
        test_folder  = os.path.join(kaggle_path, "Arabic Character Dataset", "Test Arabic")

        print("📁 Train folder:", train_folder)
        print("📁 Test folder:", test_folder)

        X_train_ext, y_train_ext = load_external_images(train_folder, target_size=self.image_size)
        X_test_ext, y_test_ext   = load_external_images(test_folder, target_size=self.image_size)

        print(f"📦 External train images: {X_train_ext.shape}, labels: {y_train_ext.shape}")
        print(f"📦 External test images: {X_test_ext.shape}, labels: {y_test_ext.shape}")

        # دمج التدريب: (إيماءات + خارجية)
        X = np.concatenate([X_gestures, X_train_ext], axis=0)
        y = np.concatenate([y_gestures, y_train_ext], axis=0)

        # دمج الاختبار
        X_test = X_test_ext
        y_test = y_test_ext

        print(f"\n🟢 Combined Training Images: {X.shape}")
        print(f"🔵 Combined Test Images: {X_test.shape}")

        return X, y, X_test, y_test

    # ----------------------------------------------------------
    #                      بناء CNN
    # ----------------------------------------------------------
    def build_cnn(self):
        H, W = self.image_size
        C = 1

        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(H, W, C)),
            BatchNormalization(),
            MaxPooling2D((2,2)),

            Conv2D(64, (3,3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2,2)),

            Conv2D(128, (3,3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2,2)),

            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.4),
            Dense(self.num_classes, activation='softmax')
        ])

        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        optimizer = Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        print("✅ CNN model created.")

    # ----------------------------------------------------------
    #                      التدريب
    # ----------------------------------------------------------
    def train(self, X_gestures, y_gestures, batch_size=32, epochs=25):
        # 1) تجهيز الداتا
        X, y, X_test_ext, y_test_ext = self.prepare_dataset(X_gestures, y_gestures)

        # 2) ترميز النصوص
        y_encoded = self.le.fit_transform(y)
        self.num_classes = len(np.unique(y_encoded))

        np.save(os.path.join(self.model_dir, "label_classes.npy"), self.le.classes_)

        y_cat = to_categorical(y_encoded, self.num_classes)

        # 3) تقسيم Train/Test (20%)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_cat, test_size=0.20, random_state=42, stratify=y_encoded
        )

        # إضافة صور الاختبار الخارجية
        if X_test_ext.size > 0:
            y_test_ext_encoded = self.le.transform(y_test_ext)
            y_test_ext_cat     = to_categorical(y_test_ext_encoded, self.num_classes)

            X_test = np.concatenate([X_test, X_test_ext], axis=0)
            y_test = np.concatenate([y_test, y_test_ext_cat], axis=0)

        print(f"\n🟢 Final Train: {X_train.shape} | Final Test: {X_test.shape}")

        # 4) بناء النموذج
        self.build_cnn()

        # 5) Callback
        # callbacks = [
        #     EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        #     ModelCheckpoint(os.path.join(self.model_dir, "best_model.h5"), monitor='val_loss', save_best_only=True)
        # ]
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
            ModelCheckpoint(os.path.join(self.model_dir, "best_model.h5"), monitor='val_loss', save_best_only=True)
        ]

        # 6) التدريب
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.1
        )
        datagen.fit(X_train)

        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_test, y_test),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        # history = self.model.fit(
        #     X_train, y_train,
        #     validation_split=0.1,
        #     epochs=epochs,
        #     batch_size=batch_size,
        #     callbacks=callbacks,
        #     verbose=1
        # )

        # حفظ
        self.model.save(os.path.join(self.model_dir, "last_model.h5"))

        # 7) التقييم
        preds = self.model.predict(X_test)
        y_true = np.argmax(y_test, axis=1)
        y_pred = np.argmax(preds, axis=1)
        acc = accuracy_score(y_true, y_pred)

        print(f"\n📊 Final Test Accuracy: {acc*100:.2f}%")

        # مصفوفة الالتباس
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=self.le.classes_)
        disp.plot(xticks_rotation=90, cmap='Blues')
        plt.title("Confusion Matrix")
        plt.show()

        return self.model, history

