import pickle
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes, filename="confusion_matrix_test.png"):
    """رسم مصفوفة الالتباس مع تحسينات"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, 
                cmap='Blues', cbar=True, annot_kws={"size": 10})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix - Test Set', fontsize=14, pad=20)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Confusion matrix saved as {filename}")

def plot_prediction_distribution(y_true, y_pred, classes, filename="prediction_distribution.png"):
    """رسم توزيع التنبؤات مقابل التسميات الحقيقية"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # التوزيع الحقيقي
    true_counts = [np.sum(y_true == i) for i in range(len(classes))]
    ax1.bar(classes, true_counts, color='skyblue', alpha=0.7, label='True')
    ax1.set_title('True Labels Distribution')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # التوزيع المتوقع
    pred_counts = [np.sum(y_pred == i) for i in range(len(classes))]
    ax2.bar(classes, pred_counts, color='lightcoral', alpha=0.7, label='Predicted')
    ax2.set_title('Predicted Labels Distribution')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Prediction distribution saved as {filename}")

def apply_consistent_scaling(X_test, scaler):
    """
    ✅ تطبيق المعايرة بنفس طريقة التدريب بالضبط
    هذه نسخة طبق الأصل من الطريقة المستخدمة في training_pipeline
    """
    print("\n🔧 Applying consistent scaling (same as training)...")
    
    # نفس الطريقة المستخدمة في training_pipeline
    n_samples, timesteps, n_features = X_test.shape
    
    # reshape البيانات بنفس طريقة التدريب
    X_test_flat = X_test.reshape(-1, n_features)
    
    print(f"📊 Before scaling - Shape: {X_test.shape}")
    print(f"📊 Before scaling - Range: [{X_test.min():.4f}, {X_test.max():.4f}]")
    print(f"📊 Before scaling - Mean: {X_test.mean():.4f}, Std: {X_test.std():.4f}")
    
    # تطبيق المعايرة بنفس الطريقة
    X_test_scaled_flat = scaler.transform(X_test_flat)
    X_test_scaled = X_test_scaled_flat.reshape(X_test.shape)
    
    print(f"📊 After scaling - Range: [{X_test_scaled.min():.4f}, {X_test_scaled.max():.4f}]")
    print(f"📊 After scaling - Mean: {X_test_scaled.mean():.4f}, Std: {X_test_scaled.std():.4f}")
    
    # فحص إذا كانت النتائج معقولة
    if abs(X_test_scaled.mean()) > 5 or X_test_scaled.std() > 5:
        print("⚠️  Warning: Scaled data statistics are higher than expected")
        print("💡 This might indicate data distribution shift between train and test")
    
    return X_test_scaled

def analyze_predictions(y_true, y_pred, y_pred_proba, classes):
    """تحليل مفصل للتنبؤات"""
    print("\n" + "="*50)
    print("🔍 PREDICTION ANALYSIS")
    print("="*50)
    
    # توزيع التنبؤات
    unique_pred, pred_counts = np.unique(y_pred, return_counts=True)
    pred_dist = dict(zip(unique_pred, pred_counts))
    
    print(f"📊 Prediction distribution: {pred_dist}")
    
    # اكتشاف إذا كان النموذج يتوقع أصنافاً محدودة
    if len(unique_pred) <= 2:
        print("🚨 CRITICAL: Model is predicting only 1-2 classes!")
        predicted_classes = [classes[i] for i in unique_pred]
        print(f"   Predicted classes: {predicted_classes}")
    elif len(unique_pred) < len(classes) // 2:
        print("⚠️  WARNING: Model is predicting only few classes")
    
    # تحليل الثقة في التنبؤات
    confidence_scores = np.max(y_pred_proba, axis=1)
    avg_confidence = np.mean(confidence_scores)
    low_confidence_threshold = 0.6
    low_confidence_count = np.sum(confidence_scores < low_confidence_threshold)
    
    print(f"📈 Average prediction confidence: {avg_confidence:.3f}")
    print(f"📉 Samples with confidence < {low_confidence_threshold}: {low_confidence_count}/{len(y_pred)} ({low_confidence_count/len(y_pred)*100:.1f}%)")
    
    # تحليل كل صنف
    print("\n📋 Per-class analysis:")
    class_results = []
    for i, class_name in enumerate(classes):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(y_pred[class_mask] == i)
            class_confidence = np.mean(confidence_scores[class_mask])
            class_samples = np.sum(class_mask)
            class_results.append({
                'class': class_name,
                'accuracy': class_accuracy,
                'confidence': class_confidence,
                'samples': class_samples
            })
            print(f"   {class_name}: Accuracy={class_accuracy:.3f}, Confidence={class_confidence:.3f}, Samples={class_samples}")
    
    return class_results

def save_detailed_results(y_true, y_pred, y_pred_proba, classes, filename="test_results_detailed.csv"):
    """حفظ النتائج المفصلة للتحليل"""
    results = []
    for i, (true, pred, proba) in enumerate(zip(y_true, y_pred, y_pred_proba)):
        results.append({
            'sample_id': i,
            'true_label': classes[true],
            'true_label_idx': true,
            'predicted_label': classes[pred],
            'predicted_label_idx': pred,
            'confidence': proba[pred],
            'is_correct': true == pred,
            'max_probability': np.max(proba),
            'entropy': -np.sum(proba * np.log(proba + 1e-8)),  # قياس عدم اليقين
            **{f'prob_{cls}': prob for cls, prob in zip(classes, proba)}
        })
    
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"✅ Detailed results saved to {filename}")
    
    # ملخص النتائج
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n📈 Overall Accuracy: {accuracy:.3f}")
    print(f"📊 Correct predictions: {df['is_correct'].sum()}/{len(df)}")
    
    return df

def check_model_compatibility(model, X_test_scaled, label_encoder):
    """فحص توافق النموذج مع البيانات"""
    print("\n🔍 Checking model compatibility...")
    
    # فحص شكل الإدخال
    expected_input_shape = model.input_shape[1:]  # (150, 21)
    actual_input_shape = X_test_scaled.shape[1:]  # (150, 21)
    
    print(f"📐 Model expects input shape: {expected_input_shape}")
    print(f"📐 Actual test data shape: {actual_input_shape}")
    
    if expected_input_shape != actual_input_shape:
        print(f"❌ Shape mismatch! Model: {expected_input_shape}, Data: {actual_input_shape}")
        return False
    
    # فحص عدد الأصناف
    expected_output_classes = model.output_shape[-1]
    actual_classes = len(label_encoder.classes_)
    
    print(f"🎯 Model expects {expected_output_classes} output classes")
    print(f"🎯 Actual number of classes: {actual_classes}")
    
    if expected_output_classes != actual_classes:
        print(f"❌ Class count mismatch! Model: {expected_output_classes}, Data: {actual_classes}")
        return False
    
    print("✅ Model and data are compatible!")
    return True

def main():
    print("🚀 Starting Enhanced Model Testing with Consistent Scaling...")
    print("="*60)
    
    try:
        # 1️⃣ تحميل بيانات الاختبار
        print("📥 Loading test data...")
        try:
            with open("X_test_fixed.pkl", "rb") as f:
                X_test = pickle.load(f)
            print("✅ Using fixed test data")
        except:
            with open("X_test.pkl", "rb") as f:
                X_test = pickle.load(f)
            print("⚠️  Using original test data")
        with open("y_test.pkl", "rb") as f:
            y_test = pickle.load(f)
        print(f"✅ Loaded X_test: {X_test.shape}, y_test: {y_test.shape}")

        # 2️⃣ تحميل Scaler و LabelEncoder
        print("📥 Loading preprocessing objects...")
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        print(f"✅ Loaded scaler and label encoder")
        print(f"🔠 Classes: {label_encoder.classes_}")

        # 3️⃣ ✅ تطبيق المعايرة بنفس طريقة التدريب بالضبط
        X_test_scaled = apply_consistent_scaling(X_test, scaler)

        # 4️⃣ تحميل النموذج المدرب
        print("\n📥 Loading trained model...")
        model_files = [
            "arabic_gesture_cnn_best.h5",
            "arabic_gesture_cnn_final.h5", 
            "model.h5"
        ]
        
        model = None
        for model_file in model_files:
            if os.path.exists(model_file):
                model = load_model(model_file)
                print(f"✅ Loaded model: {model_file}")
                break
        
        if model is None:
            raise FileNotFoundError("❌ No model file found! Please train the model first.")

        # 5️⃣ ✅ فحص توافق النموذج مع البيانات
        if not check_model_compatibility(model, X_test_scaled, label_encoder):
            print("❌ Model and data are incompatible. Please retrain the model.")
            return

        # 6️⃣ التنبؤ مع الحصول على احتمالات
        print("\n🎯 Making predictions...")
        y_pred_proba = model.predict(X_test_scaled, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        print(f"✅ Predictions completed. Shape: {y_pred.shape}")

        # 7️⃣ تحليل مفصل للتنبؤات
        class_results = analyze_predictions(y_test, y_pred, y_pred_proba, label_encoder.classes_)

        # 8️⃣ تقرير التصنيف المفصل
        print("\n" + "="*50)
        print("📊 DETAILED CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(y_test, y_pred, 
                                  target_names=label_encoder.classes_, 
                                  zero_division=0,
                                  digits=3))

        # 9️⃣ حفظ النتائج المفصلة
        results_df = save_detailed_results(y_test, y_pred, y_pred_proba, label_encoder.classes_)

        # 🔟 الرسوم البيانية
        plot_confusion_matrix(y_test, y_pred, label_encoder.classes_)
        plot_prediction_distribution(y_test, y_pred, label_encoder.classes_)

        # 🔄 تحليل إضافي للأصناف ذات الأداء الضعيف
        poor_performers = [cr for cr in class_results if cr['accuracy'] < 0.5]
        if poor_performers:
            print("\n⚠️  POOR PERFORMING CLASSES (Accuracy < 50%):")
            for cr in poor_performers:
                print(f"   {cr['class']}: {cr['accuracy']:.1%} accuracy")

        # 🎉 ملخص النتائج
        print("\n" + "="*50)
        print("🎉 TESTING COMPLETED SUCCESSFULLY!")
        print("="*50)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"🏆 Final Test Accuracy: {accuracy:.3f}")
        
        # تقدير بناء على الدقة
        if accuracy >= 0.9:
            print("🌟 EXCELLENT: Model is ready for deployment!")
        elif accuracy >= 0.8:
            print("✅ VERY GOOD: Model performance is strong")
        elif accuracy >= 0.7:
            print("📗 GOOD: Model is acceptable but can be improved") 
        elif accuracy >= 0.6:
            print("📙 FAIR: Model needs improvement")
        else:
            print("📕 POOR: Model requires significant improvements")
            
        print(f"\n📁 Results saved in:")
        print(f"   - test_results_detailed.csv")
        print(f"   - confusion_matrix_test.png") 
        print(f"   - prediction_distribution.png")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# import pickle
# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import load_model
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# def plot_confusion_matrix(y_true, y_pred, classes, filename="confusion_matrix_test.png"):
#     """رسم مصفوفة الالتباس مع تحسينات"""
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, 
#                 cmap='Blues', cbar=True, annot_kws={"size": 12})
#     plt.xlabel('Predicted', fontsize=12)
#     plt.ylabel('Actual', fontsize=12)
#     plt.title('Confusion Matrix - Test Set', fontsize=14, pad=20)
#     plt.xticks(rotation=45)
#     plt.yticks(rotation=0)
#     plt.tight_layout()
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     plt.show()
#     print(f"✅ Confusion matrix saved as {filename}")

# def plot_prediction_distribution(y_true, y_pred, classes, filename="prediction_distribution.png"):
#     """رسم توزيع التنبؤات مقابل التسميات الحقيقية"""
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
#     # التوزيع الحقيقي
#     true_counts = [np.sum(y_true == i) for i in range(len(classes))]
#     ax1.bar(classes, true_counts, color='skyblue', alpha=0.7, label='True')
#     ax1.set_title('True Labels Distribution')
#     ax1.set_xlabel('Class')
#     ax1.set_ylabel('Count')
#     ax1.tick_params(axis='x', rotation=45)
    
#     # التوزيع المتوقع
#     pred_counts = [np.sum(y_pred == i) for i in range(len(classes))]
#     ax2.bar(classes, pred_counts, color='lightcoral', alpha=0.7, label='Predicted')
#     ax2.set_title('Predicted Labels Distribution')
#     ax2.set_xlabel('Class')
#     ax2.set_ylabel('Count')
#     ax2.tick_params(axis='x', rotation=45)
    
#     plt.tight_layout()
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     plt.show()
#     print(f"✅ Prediction distribution saved as {filename}")

# def analyze_predictions(y_true, y_pred, y_pred_proba, classes):
#     """تحليل مفصل للتنبؤات"""
#     print("\n" + "="*50)
#     print("🔍 PREDICTION ANALYSIS")
#     print("="*50)
    
#     # توزيع التنبؤات
#     unique_pred, pred_counts = np.unique(y_pred, return_counts=True)
#     pred_dist = dict(zip(unique_pred, pred_counts))
    
#     print(f"📊 Prediction distribution: {pred_dist}")
    
#     # اكتشاف إذا كان النموذج يتوقع صنفاً واحداً فقط
#     if len(unique_pred) == 1:
#         print("🚨 CRITICAL: Model is predicting only ONE class!")
#         print(f"   All predictions are for class: {classes[unique_pred[0]]}")
#     elif len(unique_pred) < len(classes) // 2:
#         print("⚠️  WARNING: Model is predicting only few classes")
    
#     # تحليل الثقة في التنبؤات
#     confidence_scores = np.max(y_pred_proba, axis=1)
#     avg_confidence = np.mean(confidence_scores)
#     low_confidence_threshold = 0.6
#     low_confidence_count = np.sum(confidence_scores < low_confidence_threshold)
    
#     print(f"📈 Average prediction confidence: {avg_confidence:.3f}")
#     print(f"📉 Samples with confidence < {low_confidence_threshold}: {low_confidence_count}/{len(y_pred)} ({low_confidence_count/len(y_pred)*100:.1f}%)")
    
#     # تحليل كل صنف
#     print("\n📋 Per-class analysis:")
#     for i, class_name in enumerate(classes):
#         class_mask = y_true == i
#         if np.sum(class_mask) > 0:
#             class_accuracy = np.mean(y_pred[class_mask] == i)
#             class_confidence = np.mean(confidence_scores[class_mask])
#             print(f"   {class_name}: Accuracy={class_accuracy:.3f}, Avg Confidence={class_confidence:.3f}")

# def validate_data_quality(X_test, y_test, scaler):
#     """فحص جودة البيانات قبل الاختبار"""
#     print("\n" + "="*50)
#     print("🔍 DATA QUALITY CHECK")
#     print("="*50)
    
#     # فحص البيانات الأصلية
#     print(f"📊 X_test shape: {X_test.shape}")
#     print(f"📊 y_test shape: {y_test.shape}")
#     print(f"🎯 Unique classes in y_test: {np.unique(y_test)}")
#     print(f"📈 Class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
#     # فحص القيم غير الطبيعية
#     print(f"🔍 Data stats - Min: {X_test.min():.4f}, Max: {X_test.max():.4f}")
#     print(f"🔍 Data stats - Mean: {X_test.mean():.4f}, Std: {X_test.std():.4f}")
#     print(f"🔍 NaN values: {np.isnan(X_test).sum()}, Inf values: {np.isinf(X_test).sum()}")
    
#     # فحص المعايرة
#     X_test_flat = X_test.reshape(-1, X_test.shape[-1])
#     X_test_scaled_flat = scaler.transform(X_test_flat)
    
#     print(f"📊 After scaling - Min: {X_test_scaled_flat.min():.4f}, Max: {X_test_scaled_flat.max():.4f}")
#     print(f"📊 After scaling - Mean: {X_test_scaled_flat.mean():.4f}, Std: {X_test_scaled_flat.std():.4f}")
    
#     # تحذير إذا كانت البيانات بعد المعايرة غير طبيعية
#     if abs(X_test_scaled_flat.mean()) > 10 or X_test_scaled_flat.std() > 10:
#         print("⚠️  WARNING: Scaled data has unusual statistics!")
    
#     return X_test_scaled_flat.reshape(X_test.shape)

# def save_detailed_results(y_true, y_pred, y_pred_proba, classes, filename="test_results_detailed.csv"):
#     """حفظ النتائج المفصلة للتحليل"""
#     results = []
#     for i, (true, pred, proba) in enumerate(zip(y_true, y_pred, y_pred_proba)):
#         results.append({
#             'sample_id': i,
#             'true_label': classes[true],
#             'true_label_idx': true,
#             'predicted_label': classes[pred],
#             'predicted_label_idx': pred,
#             'confidence': proba[pred],
#             'is_correct': true == pred,
#             'max_probability': np.max(proba),
#             'entropy': -np.sum(proba * np.log(proba + 1e-8))  # قياس عدم اليقين
#         })
    
#     df = pd.DataFrame(results)
#     df.to_csv(filename, index=False, encoding='utf-8-sig')
#     print(f"✅ Detailed results saved to {filename}")
    
#     # ملخص النتائج
#     accuracy = accuracy_score(y_true, y_pred)
#     print(f"\n📈 Overall Accuracy: {accuracy:.3f}")
#     print(f"📊 Correct predictions: {df['is_correct'].sum()}/{len(df)}")
    
#     return df

# def main():
#     print("🚀 Starting Enhanced Model Testing...")
#     print("="*60)
    
#     try:
#         # 1️⃣ تحميل بيانات الاختبار
#         print("📥 Loading test data...")
#         with open("X_test.pkl", "rb") as f:
#             X_test = pickle.load(f)
#         with open("y_test.pkl", "rb") as f:
#             y_test = pickle.load(f)
#         print(f"✅ Loaded X_test: {X_test.shape}, y_test: {y_test.shape}")

#         # 2️⃣ تحميل Scaler و LabelEncoder
#         print("📥 Loading preprocessing objects...")
#         with open("scaler.pkl", "rb") as f:
#             scaler = pickle.load(f)
#         with open("label_encoder.pkl", "rb") as f:
#             label_encoder = pickle.load(f)
#         print(f"✅ Loaded scaler and label encoder")
#         print(f"🔠 Classes: {label_encoder.classes_}")

#         # 3️⃣ فحص جودة البيانات وتطبيق المعايرة
#         X_test_scaled = validate_data_quality(X_test, y_test, scaler)

#         # 4️⃣ تحميل النموذج المدرب
#         print("\n📥 Loading trained model...")
#         if os.path.exists("arabic_gesture_cnn_best.h5"):
#             model = load_model("arabic_gesture_cnn_best.h5")
#             print("✅ Loaded model: arabic_gesture_cnn_best.h5")
#         elif os.path.exists("arabic_gesture_cnn_final.h5"):
#             model = load_model("arabic_gesture_cnn_final.h5")
#             print("✅ Loaded model: arabic_gesture_cnn_final.h5")
#         else:
#             raise FileNotFoundError("❌ No model file found!")

#         # عرض ملخص النموذج
#         print(f"✅ Model input shape: {model.input_shape}")
#         print(f"✅ Model output shape: {model.output_shape}")

#         # 5️⃣ التنبؤ مع الحصول على احتمالات
#         print("\n🎯 Making predictions...")
#         y_pred_proba = model.predict(X_test_scaled, verbose=1)
#         y_pred = np.argmax(y_pred_proba, axis=1)
#         print(f"✅ Predictions completed. Shape: {y_pred.shape}")

#         # 6️⃣ تحليل مفصل للتنبؤات
#         analyze_predictions(y_test, y_pred, y_pred_proba, label_encoder.classes_)

#         # 7️⃣ تقرير التصنيف المفصل
#         print("\n" + "="*50)
#         print("📊 DETAILED CLASSIFICATION REPORT")
#         print("="*50)
#         print(classification_report(y_test, y_pred, 
#                                   target_names=label_encoder.classes_, 
#                                   zero_division=0,
#                                   digits=3))

#         # 8️⃣ حفظ النتائج المفصلة
#         results_df = save_detailed_results(y_test, y_pred, y_pred_proba, label_encoder.classes_)

#         # 9️⃣ الرسوم البيانية
#         plot_confusion_matrix(y_test, y_pred, label_encoder.classes_)
#         plot_prediction_distribution(y_test, y_pred, label_encoder.classes_)

#         # 🔟 ملخص النتائج
#         print("\n" + "="*50)
#         print("🎉 TESTING COMPLETED SUCCESSFULLY!")
#         print("="*50)
#         accuracy = accuracy_score(y_test, y_pred)
#         print(f"🏆 Final Test Accuracy: {accuracy:.3f}")
#         print(f"📁 Results saved in:")
#         print(f"   - test_results_detailed.csv")
#         print(f"   - confusion_matrix_test.png") 
#         print(f"   - prediction_distribution.png")
        
#         # نصيحة بناء على النتائج
#         if accuracy < 0.5:
#             print("\n💡 RECOMMENDATION: Model performance is low. Consider:")
#             print("   - Retraining with better data preprocessing")
#             print("   - Checking for class imbalance")
#             print("   - Verifying feature extraction")
#         elif accuracy < 0.8:
#             print("\n💡 RECOMMENDATION: Good performance. Can be improved with:")
#             print("   - More training data")
#             print("   - Hyperparameter tuning")
#             print("   - Data augmentation")
#         else:
#             print("\n💡 RECOMMENDATION: Excellent performance! Model is ready for deployment.")

#     except Exception as e:
#         print(f"❌ Error during testing: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()