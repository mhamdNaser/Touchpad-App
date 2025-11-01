# app/services/data_splitter.py
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple

class DataSplitter:
    def __init__(self, test_size: float = 0.2, validation_size: float = 0.1, random_state: int = 42):
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
    
    def split_data(self, features: np.ndarray, labels: np.ndarray) -> Tuple:
        """
        تقسيم البيانات إلى تدريب، تحقق، واختبار
        """
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, 
            test_size=self.test_size,
            stratify=labels,  
            random_state=self.random_state
        )
        
        val_size_adjusted = self.validation_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=self.random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_split_info(self, y_train, y_val, y_test) -> Dict:
        """
        الحصول على معلومات عن التقسيم
        """
        train_counts = np.unique(y_train, return_counts=True)
        val_counts = np.unique(y_val, return_counts=True)
        test_counts = np.unique(y_test, return_counts=True)
        
        return {
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test),
            'train_distribution': dict(zip(train_counts[0], train_counts[1])),
            'val_distribution': dict(zip(val_counts[0], val_counts[1])),
            'test_distribution': dict(zip(test_counts[0], test_counts[1]))
        }