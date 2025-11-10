import numpy as np

def augment_gesture(feature_matrix: np.ndarray, n_aug: int = 3) -> list:
    """
    توليد n_aug نسخة معدلة من feature_matrix.
    feature_matrix: (sequence_length, 12)
    """
    augmented = []
    for _ in range(n_aug):
        X = feature_matrix[:, 0]  # x
        Y = feature_matrix[:, 1]  # y
        Pressure = feature_matrix[:, 5]
        
        # 1. ضوضاء عشوائية
        X_new = X + np.random.normal(0, 0.5, size=X.shape)
        Y_new = Y + np.random.normal(0, 0.5, size=Y.shape)
        
        # 2. تدوير بسيط
        angle = np.random.uniform(-0.1, 0.1)
        X_rot = X_new*np.cos(angle) - Y_new*np.sin(angle)
        Y_rot = X_new*np.sin(angle) + Y_new*np.cos(angle)
        
        # 3. تغيير مقياس
        scale = np.random.uniform(0.95, 1.05)
        X_rot *= scale
        Y_rot *= scale
        
        # الاحتفاظ بالميزات الأخرى كما هي
        new_matrix = feature_matrix.copy()
        new_matrix[:, 0] = X_rot
        new_matrix[:, 1] = Y_rot
        new_matrix[:, 5] = Pressure  # يمكن تعديل الضغط إذا أحببت
        
        augmented.append(new_matrix)
    return augmented
