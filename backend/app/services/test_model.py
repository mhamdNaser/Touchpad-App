# app/test_model_simple.py
import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes, filename="confusion_matrix_test.png"):
    """رسم مصفوفة الالتباس"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"✅ Confusion matrix saved as {filename}")

def plot_prediction_distribution(y_true, y_pred, classes, filename="prediction_distribution.png"):
    """رسم توزيع التنبؤات مقابل التسميات الحقيقية"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # التوزيع الحقيقي
    true_counts = [np.sum(y_true == i) for i in range(len(classes))]
    ax1.bar(classes, true_counts, color='skyblue', alpha=0.7)
    ax1.set_title('True Labels Distribution')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # التوزيع المتوقع
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

def main():
    print("🚀 Starting Model Testing (No Scaler Required)...")

    # 1️⃣ تحميل بيانات الاختبار
    if not os.path.exists("X_test.pkl") or not os.path.exists("y_test.pkl"):
        raise FileNotFoundError("❌ X_test.pkl or y_test.pkl not found! Please run training first.")
    
    with open("X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open("y_test.pkl", "rb") as f:
        y_test = pickle.load(f)
    
    print(f"📊 Test data loaded: X_test {X_test.shape}, y_test {y_test.shape}")

    # 2️⃣ تحميل LabelEncoder
    if not os.path.exists("label_encoder.pkl"):
        raise FileNotFoundError("❌ label_encoder.pkl not found! Please run training first.")
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    print(f"🔠 Classes: {label_encoder.classes_}")

    # 3️⃣ تحميل النموذج
    model_file = "arabic_gesture_cnn_best.h5"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"❌ {model_file} not found! Please run training first.")
    
    model = load_model(model_file)
    print(f"✅ Loaded model: {model_file}")

    # 4️⃣ التنبؤ
    y_pred_proba = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 5️⃣ التقارير
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🏆 Test Accuracy: {accuracy:.3f}")
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    
    # 6️⃣ الرسوم البيانية
    plot_confusion_matrix(y_test, y_pred, label_encoder.classes_)
    plot_prediction_distribution(y_test, y_pred, label_encoder.classes_)

if __name__ == "__main__":
    main()
