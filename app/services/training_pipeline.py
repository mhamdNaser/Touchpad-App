# app/services/training_pipeline.py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

class GestureTrainer:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=self.input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X, y, epochs=30, batch_size=32):
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        y_cat = to_categorical(y_encoded, num_classes=self.num_classes)
        self.model.fit(X, y_cat, epochs=epochs, batch_size=batch_size, validation_split=0.1)
        return self.model, le
