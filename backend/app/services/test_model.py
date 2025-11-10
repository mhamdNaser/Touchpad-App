import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report

model = load_model("arabic_gesture_lstm_final.h5")

num_classes = len(np.unique(y_test))
y_test_cat = to_categorical(y_test, num_classes)

y_pred_prob = model.predict(X_test)  
y_pred = np.argmax(y_pred_prob, axis=1)  


accuracy = accuracy_score(y_test, y_pred)
print(f"✅ دقة النموذج على Test set: {accuracy:.3f}")


print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
