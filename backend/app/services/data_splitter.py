# app/services/data_splitter.py
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple, Dict

class DataSplitter:
    def __init__(self, test_size: float = 0.2, validation_size: float = 0.1, random_state: int = 42):
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
    
    def split_data(self, features: np.ndarray, labels: np.ndarray) -> Tuple:
        """
        تقسيم بيانات الإيماءات إلى تدريب، تحقق، واختبار (جاهزة لـ LSTM)
        """
        # التحقق من أن stratify ممكن
        stratify_labels = labels if len(np.unique(labels)) > 1 else None

        # تقسيم مؤقت (تدريب + تحقق) و (اختبار)
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, 
            test_size=self.test_size,
            stratify=stratify_labels,
            random_state=self.random_state
        )
        
        # حساب النسبة المعدلة للتحقق
        val_size_adjusted = self.validation_size / (1 - self.test_size)

        stratify_temp = y_temp if len(np.unique(y_temp)) > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=stratify_temp,
            random_state=self.random_state
        )

        # إرجاع المجموعات بشكل منظم
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_split_info(self, y_train, y_val, y_test) -> Dict:
        """
        إحصائيات عن توزيع البيانات بعد التقسيم
        """
        def get_distribution(y):
            unique, counts = np.unique(y, return_counts=True)
            return dict(zip(unique, counts))
        
        return {
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test),
            'train_distribution': get_distribution(y_train),
            'val_distribution': get_distribution(y_val),
            'test_distribution': get_distribution(y_test),
        }
