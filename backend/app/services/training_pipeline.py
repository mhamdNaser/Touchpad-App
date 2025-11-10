# app/services/training_pipeline.py

import json
import pickle
import numpy as np
from typing import List
from app.services.data_loader import DataLoader
from app.services.features import FeatureEngineer
from app.services.data_splitter import DataSplitter

from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


class TrainingPipeline:
    def __init__(self, db=None, max_timesteps: int = 200):
        self.db = db
        self.data_loader = DataLoader(api_url="https://api.sydev.site/api/gestures")  # API-based loader
        self.feature_engineer = FeatureEngineer(max_timesteps=max_timesteps)
        self.data_splitter = DataSplitter()

    def build_cnn_model(self, input_shape, num_classes):
        """
        نموذج 1D-CNN لاستخراج الميزات من التسلسل الزمني لكل إيماءة
        """
        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),

            Conv1D(32, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),

            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        return model

    def _convert_keys_to_int(self, d):
        """
        تحويل كل مفاتيح وقيم dict من np.int64 إلى int، بما في ذلك المفاتيح المتداخلة
        """
        result = {}
        for k, v in d.items():
            # تحويل المفتاح
            key = int(k) if isinstance(k, (np.integer, np.int64)) else k

            # تحويل القيمة
            if isinstance(v, dict):
                result[key] = self._convert_keys_to_int(v)
            elif isinstance(v, (np.integer, np.int64)):
                result[key] = int(v)
            else:
                result[key] = v
        return result


    def train_model(self, characters: List[str]):
        print("🚀 Starting 1D-CNN training pipeline...")

        # 1️⃣ تحميل البيانات من API
        gestures_data = self.data_loader.load_gestures_data(characters)
        print(f"✅ Loaded {len(gestures_data)} gestures")

        if len(gestures_data) == 0:
            raise ValueError("No gestures loaded. Check API or character list.")

        # 2️⃣ استخراج الميزات
        X, y = self.feature_engineer.extract_features(gestures_data)
        print(f"🔹 Data shape: {X.shape}, number of classes: {len(np.unique(y))}")

        # 3️⃣ تقسيم البيانات
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_splitter.split_data(X, y)
        split_info = self.data_splitter.get_split_info(y_train, y_val, y_test)

        # تصحيح مفاتيح dict قبل الطباعة
        split_info_clean = self._convert_keys_to_int(split_info)
        print(json.dumps(split_info_clean, indent=2, ensure_ascii=False))

        # 4️⃣ تحويل التسميات إلى one-hot
        num_classes = len(np.unique(y))
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)

        # 5️⃣ بناء النموذج
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self.build_cnn_model(input_shape, num_classes)
        model.summary()

        # 6️⃣ Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint(
            filepath='arabic_gesture_cnn_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )

        # 7️⃣ التدريب
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=100,
            batch_size=16,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )

        # 8️⃣ التقييم
        test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"✅ Test accuracy: {test_acc:.3f}")

        y_pred = np.argmax(model.predict(X_test), axis=1)
        print("\n📊 Classification report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # 9️⃣ حفظ النموذج و الـ Scaler و الـ LabelEncoder
        model.save("arabic_gesture_cnn_final.h5")
        with open("scaler.pkl", "wb") as f:
            pickle.dump(self.feature_engineer.scaler, f)
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(self.feature_engineer.label_encoder, f)

        return {
            'model': model,
            'test_accuracy': float(test_acc),
            'split_info': split_info_clean
        }





# # app/services/training_pipeline.py

# import json
# import pickle
# import numpy as np
# from typing import List
# from app.services.data_loader import DataLoader
# from app.services.features import FeatureEngineer
# from app.services.data_splitter import DataSplitter

# from sklearn.metrics import classification_report
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.optimizers import Adam

# class TrainingPipeline:
#     def __init__(self, db=None, max_timesteps: int = 200):
#         self.db = db
#         self.data_loader = DataLoader()  # API-based loader
#         self.feature_engineer = FeatureEngineer(max_timesteps=max_timesteps)
#         self.data_splitter = DataSplitter()

#     def build_lstm_model(self, input_shape, num_classes):
#         model = Sequential([
#             LSTM(128, input_shape=input_shape, return_sequences=True),
#             BatchNormalization(),
#             Dropout(0.3),

#             LSTM(64),
#             BatchNormalization(),
#             Dropout(0.3),

#             Dense(64, activation='relu'),
#             Dropout(0.2),

#             Dense(num_classes, activation='softmax')
#         ])
#         model.compile(
#             loss='categorical_crossentropy',
#             optimizer=Adam(learning_rate=0.001),
#             metrics=['accuracy']
#         )
#         return model

#     def train_model(self, characters: List[str]):
#         print("🚀 Starting LSTM training pipeline...")

#         # 1️⃣ تحميل البيانات من API
#         gestures_data = self.data_loader.load_gestures_data(characters)
#         print(f"✅ Loaded {len(gestures_data)} gestures")

#         if len(gestures_data) == 0:
#             raise ValueError("No gestures loaded. Check API or character list.")

#         # 2️⃣ استخراج الميزات
#         X, y = self.feature_engineer.extract_features(gestures_data)
#         print(f"🔹 Data shape: {X.shape}, number of classes: {len(np.unique(y))}")

#         # 3️⃣ تقسيم البيانات
#         X_train, X_val, X_test, y_train, y_val, y_test = self.data_splitter.split_data(X, y)
#         split_info = self.data_splitter.get_split_info(y_train, y_val, y_test)

#         # 4️⃣ تحويل التسميات إلى one-hot
#         num_classes = len(np.unique(y))
#         y_train_cat = to_categorical(y_train, num_classes)
#         y_val_cat = to_categorical(y_val, num_classes)
#         y_test_cat = to_categorical(y_test, num_classes)

#         # 5️⃣ بناء النموذج
#         input_shape = (X_train.shape[1], X_train.shape[2])
#         model = self.build_lstm_model(input_shape, num_classes)
#         model.summary()

#         # 6️⃣ Callbacks
#         early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#         checkpoint = ModelCheckpoint(
#             filepath='arabic_gesture_lstm_best.h5',
#             monitor='val_accuracy',
#             save_best_only=True,
#             verbose=1
#         )

#         # 7️⃣ التدريب
#         history = model.fit(
#             X_train, y_train_cat,
#             validation_data=(X_val, y_val_cat),
#             epochs=100,
#             batch_size=32,
#             callbacks=[early_stop, checkpoint],
#             verbose=1
#         )

#         # 8️⃣ التقييم
#         test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
#         print(f"✅ Test accuracy: {test_acc:.3f}")

#         y_pred = np.argmax(model.predict(X_test), axis=1)
#         print("\n📊 Classification report:")
#         print(classification_report(y_test, y_pred, zero_division=0))

#         # 9️⃣ حفظ النموذج والـ Scaler
#         model.save("arabic_gesture_lstm_final.h5")
#         with open("scaler.pkl", "wb") as f:
#             pickle.dump(self.feature_engineer.scaler, f)
#         with open("label_encoder.pkl", "wb") as f:
#             pickle.dump(self.feature_engineer.label_encoder, f)

#         return {
#             'model': model,
#             'test_accuracy': float(test_acc),
#             'split_info': split_info
#         }



# # app/services/training_pipeline.py

# import json
# import pickle
# import numpy as np
# from typing import List
# from app.services.data_loader import DataLoader
# from app.services.features import FeatureEngineer
# from app.services.data_splitter import DataSplitter

# from sklearn.metrics import classification_report, accuracy_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.optimizers import Adam

# class TrainingPipeline:
#     def __init__(self, db):
#         self.db = db
#         self.data_loader = DataLoader(db)
#         self.feature_engineer = FeatureEngineer(max_timesteps=100)
#         self.data_splitter = DataSplitter()

#     def build_lstm_model(self, input_shape, num_classes):
#         model = Sequential([
#             LSTM(128, input_shape=input_shape, return_sequences=True),
#             BatchNormalization(),
#             Dropout(0.3),

#             LSTM(64),
#             BatchNormalization(),
#             Dropout(0.3),

#             Dense(64, activation='relu'),
#             Dropout(0.2),

#             Dense(num_classes, activation='softmax')
#         ])

#         model.compile(
#             loss='categorical_crossentropy',
#             optimizer=Adam(learning_rate=0.001),
#             metrics=['accuracy']
#         )
#         return model

#     def train_model(self, characters: List[str]):
#         print("Starting LSTM training pipeline...")

#         gestures_data = self.data_loader.load_gestures_data(characters)
#         print(f"Loaded {len(gestures_data)} gestures")

#         X, y = self.feature_engineer.extract_features(gestures_data)
#         print(f"Data shape: {X.shape}, number of classes: {len(np.unique(y))}")

#         X_train, X_val, X_test, y_train, y_val, y_test = self.data_splitter.split_data(X, y)
#         split_info = self._convert_split_info_to_int(
#             self.data_splitter.get_split_info(y_train, y_val, y_test)
#         )
#         print(json.dumps(split_info, indent=2, ensure_ascii=False))

#         num_classes = len(np.unique(y))
#         y_train_cat = to_categorical(y_train, num_classes)
#         y_val_cat = to_categorical(y_val, num_classes)
#         y_test_cat = to_categorical(y_test, num_classes)

#         input_shape = (X_train.shape[1], X_train.shape[2])
#         model = self.build_lstm_model(input_shape, num_classes)
#         model.summary()

#         early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#         checkpoint = ModelCheckpoint(
#             filepath='arabic_gesture_lstm_best.h5',
#             monitor='val_accuracy',
#             save_best_only=True,
#             verbose=1
#         )

#         history = model.fit(
#             X_train, y_train_cat,
#             validation_data=(X_val, y_val_cat),
#             epochs=100,
#             batch_size=32,
#             callbacks=[early_stop, checkpoint],
#             verbose=1
#         )

#         test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
#         print(f"Test accuracy: {test_acc:.3f}")

#         y_pred = np.argmax(model.predict(X_test), axis=1)
#         print(classification_report(y_test, y_pred, zero_division=0))

#         model.save("arabic_gesture_lstm_final.h5")

#         with open("scaler.pkl", "wb") as f:
#             pickle.dump(self.feature_engineer.scaler, f)

#         return {
#             'model': model,
#             'test_accuracy': float(test_acc),
#             'split_info': split_info
#         }

#     def _convert_split_info_to_int(self, split_info: dict) -> dict:
#         return {
#             'train_samples': split_info['train_samples'],
#             'val_samples': split_info['val_samples'],
#             'test_samples': split_info['test_samples'],
#             'train_distribution': {int(k): int(v) for k, v in split_info['train_distribution'].items()},
#             'val_distribution': {int(k): int(v) for k, v in split_info['val_distribution'].items()},
#             'test_distribution': {int(k): int(v) for k, v in split_info['test_distribution'].items()}
#         }
