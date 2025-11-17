# app/test_model_simple.py

import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# =======================================
# رسم مصفوفة الالتباس
# =======================================
def plot_confusion_matrix(y_true, y_pred, classes, filename="confusion_matrix_test.png"):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes,
                yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"✅ Confusion matrix saved as {filename}")


# =======================================
# رسم توزيع التوقعات والحقيقة
# =======================================
def plot_prediction_distribution(y_true, y_pred, classes, filename="prediction_distribution.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    true_counts = [np.sum(y_true == i) for i in range(len(classes))]
    ax1.bar(classes, true_counts, color='skyblue', alpha=0.7)
    ax1.set_title('True Labels Distribution')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)

    pred_counts = [np.sum(y_pred == i) for i in range(len(classes))]
    ax2.bar(classes, pred_counts, color='lightcoral', alpha=0.7)
    ax2.set_title('Predicted Labels Distribution')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"✅ Prediction distribution saved as {filename}")


# =======================================
# Main Testing Script
# =======================================
def main():
    print("🚀 Starting Model Testing ...")

    # -------------------------
    # 1) Load test datasets
    # -------------------------
    if not os.path.exists("X_test.pkl") or not os.path.exists("y_test.pkl"):
        raise FileNotFoundError("❌ X_test.pkl or y_test.pkl not found!")

    with open("X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open("y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    print(f"📊 X_test shape: {X_test.shape}")
    print(f"📊 y_test shape: {y_test.shape}")

    timesteps = X_test.shape[1]
    feature_dim = X_test.shape[2]
    print(f"🔍 Each sample = {timesteps} timesteps × {feature_dim} features")

    # -------------------------
    # 2) Load label encoder
    # -------------------------
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    classes = label_encoder.classes_
    print(f"🔠 Classes: {classes}")

    # -------------------------
    # 3) Load final model
    # -------------------------
    model_file = "arabic_gesture_cnn_final.h5"   # الأفضل للاختبار
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"❌ {model_file} not found!")

    model = load_model(model_file)
    print(f"✅ Loaded model: {model_file}")

    # -------------------------
    # 4) Predict
    # -------------------------
    print("🔮 Running predictions...")
    y_pred_proba = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # -------------------------
    # 5) Evaluation
    # -------------------------
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🏆 Test Accuracy = {acc:.3f}")

    print("\n📊 Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=classes,
        zero_division=0
    ))

    # -------------------------
    # 6) Confusion matrix & distributions
    # -------------------------
    plot_confusion_matrix(y_test, y_pred, classes)
    plot_prediction_distribution(y_test, y_pred, classes)

    # -------------------------
    # 7) Show real vs predicted characters
    # -------------------------
    print("\n🔍 Example Predictions (real → predicted):")
    for i in range(min(20, len(y_test))):
        real_char = classes[y_test[i]]
        pred_char = classes[y_pred[i]]
        print(f"{i+1:02d}) {real_char} → {pred_char}")


if __name__ == "__main__":
    main()
