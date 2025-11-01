# app/services/training_pipeline.py
from app.services.data_loader import DataLoader
from app.services.features import FeatureEngineer
from app.services.data_splitter import DataSplitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

class TrainingPipeline:
    def __init__(self, db):
        self.db = db
        self.data_loader = DataLoader(db)
        self.feature_engineer = FeatureEngineer()
        self.data_splitter = DataSplitter()
    
    def train_model(self, characters: List[str]):
        """
        خطوات التدريب الكاملة
        """
        print("🎯 بداية تدريب النموذج...")
        
        # 1. تحميل البيانات
        print("📥 جلب البيانات من قاعدة البيانات...")
        gestures_data = self.data_loader.load_gestures_data(characters)
        print(f"تم تحميل {len(gestures_data)} إيماءة")
        
        # 2. استخراج الميزات
        print("🔧 استخراج الميزات...")
        features, labels = self.feature_engineer.extract_features(gestures_data)
        print(f"الأبعاد: {features.shape}")
        
        # 3. تقسيم البيانات
        print("📊 تقسيم البيانات...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_splitter.split_data(features, labels)
        
        split_info = self.data_splitter.get_split_info(y_train, y_val, y_test)
        print("معلومات التقسيم:")
        print(json.dumps(split_info, indent=2, ensure_ascii=False))
        
        # 4. تدريب النموذج (مثال باستخدام RandomForest)
        print("🤖 تدريب النموذج...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 5. التقيم
        print("📈 تقييم النموذج...")
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        
        print(f"دقة التدريب: {train_accuracy:.3f}")
        print(f"دقة التحقق: {val_accuracy:.3f}")
        
        # 6. حفظ النموذج
        print("💾 حفظ النموذج...")
        joblib.dump(model, 'arabic_gesture_model.pkl')
        
        return {
            'model': model,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'split_info': split_info
        }