import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class FeatureEngineerVisualizer:
    def __init__(self, max_timesteps: int = 150, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        self.scaler = RobustScaler()  # ✅ تغيير من StandardScaler إلى RobustScaler
        self.label_encoder = LabelEncoder()
        self.max_timesteps = max_timesteps
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    # ======================================================
    # 🧠 استخراج الميزات من كل حركة (Gesture) - مُحسّن
    # ======================================================
    def extract_sequence_features(self, gesture: Dict) -> np.ndarray:
        frames = gesture.get('frames', [])
        
        # إذا كانت البيانات في الصيغة القديمة
        if not frames and 'points' in gesture:
            frames = [{'points': gesture['points'], 'delta_ms': gesture.get('delta_ms', 16)}]  # 16ms افتراضي

        if not frames:
            return None

        sequence = []
        prev_points = None

        for i, frame in enumerate(frames):
            points = frame.get('points', [])
            if not points:
                continue

            delta_ms = max(frame.get('delta_ms', 16), 1)  # ✅ تجنب القسمة على صفر
            delta_s = delta_ms / 1000.0

            # استخراج الإحداثيات الأساسية
            x = np.array([p.get('x', 0) for p in points])
            y = np.array([p.get('y', 0) for p in points])
            pressure = np.array([p.get('pressure', 0.5) for p in points])  # قيمة افتراضية 0.5
            angle = np.array([p.get('angle', 0) for p in points])

            # ✅ حساب السرعة والتسارع بشكل أكثر دقة
            if i == 0 or prev_points is None:
                # الإطار الأول - استخدام قيم صفرية
                vx = np.zeros_like(x)
                vy = np.zeros_like(y)
                ax = np.zeros_like(x)
                ay = np.zeros_like(y)
            else:
                # حساب السرعة بناء على التغير في المواقع
                prev_x = np.array([p.get('x', 0) for p in prev_points])
                prev_y = np.array([p.get('y', 0) for p in prev_points])
                
                dx = x - prev_x
                dy = y - prev_y
                
                vx = dx / delta_s
                vy = dy / delta_s
                
                # ✅ تحسين حساب التسارع
                if i == 1:
                    ax = np.zeros_like(vx)
                    ay = np.zeros_like(vy)
                else:
                    # نحتاج لحساب السرعة السابقة
                    prev_prev_points = frames[i-2].get('points', [])
                    if prev_prev_points:
                        prev_prev_x = np.array([p.get('x', 0) for p in prev_prev_points])
                        prev_prev_y = np.array([p.get('y', 0) for p in prev_prev_points])
                        prev_delta_ms = max(frames[i-1].get('delta_ms', 16), 1)
                        prev_delta_s = prev_delta_ms / 1000.0
                        
                        prev_dx = prev_x - prev_prev_x
                        prev_dy = prev_y - prev_prev_y
                        prev_vx = prev_dx / prev_delta_s
                        prev_vy = prev_dy / prev_delta_s
                        
                        ax = (vx - prev_vx) / delta_s
                        ay = (vy - prev_vy) / delta_s
                    else:
                        ax = np.zeros_like(vx)
                        ay = np.zeros_like(vy)

            # magnitude محسوب بشكل صحيح
            velocity_magnitude = np.sqrt(vx**2 + vy**2)
            acceleration_magnitude = np.sqrt(ax**2 + ay**2)

            # ✅ ميزات محسنة مع قيم افتراضية آمنة
            frame_features = [
                np.mean(x) if len(x) > 0 else 0.0,
                np.std(x) if len(x) > 0 else 0.1,
                np.mean(y) if len(y) > 0 else 0.0,
                np.std(y) if len(y) > 0 else 0.1,
                np.mean(pressure) if len(pressure) > 0 else 0.5,
                np.std(pressure) if len(pressure) > 0 else 0.1,
                np.mean(angle) if len(angle) > 0 else 0.0,
                np.std(angle) if len(angle) > 0 else 0.1,
                np.mean(vx) if len(vx) > 0 else 0.0,
                np.std(vx) if len(vx) > 0 else 0.1,
                np.mean(vy) if len(vy) > 0 else 0.0,
                np.std(vy) if len(vy) > 0 else 0.1,
                np.mean(ax) if len(ax) > 0 else 0.0,
                np.std(ax) if len(ax) > 0 else 0.1,
                np.mean(ay) if len(ay) > 0 else 0.0,
                np.std(ay) if len(ay) > 0 else 0.1,
                np.mean(velocity_magnitude) if len(velocity_magnitude) > 0 else 0.0,
                np.std(velocity_magnitude) if len(velocity_magnitude) > 0 else 0.1,
                np.mean(acceleration_magnitude) if len(acceleration_magnitude) > 0 else 0.0,
                delta_s,
                len(points)
            ]
            
            sequence.append(frame_features)
            prev_points = points  # حفظ النقاط للحسابات التالية

        if not sequence:
            return None

        # ✅ تطبيع الطول مع حشو ذكي (لا أصفار)
        feature_dim = len(sequence[0])
        if len(sequence) < self.max_timesteps:
            # استخدام متوسط الإطارات الأخيرة للحشو بدلاً من الأصفار
            last_frame = sequence[-1]
            padding_frames = [last_frame] * (self.max_timesteps - len(sequence))
            sequence.extend(padding_frames)
        else:
            sequence = sequence[:self.max_timesteps]

        return np.array(sequence)

    # ======================================================
    # 🔄 تقسيم البيانات + السكايل + التشفير - مُحسّن
    # ======================================================
    def split_data(self, gestures_data: List[Dict], fixed_indices=None):
        features, labels = [], []
        
        print(f"🔍 Processing {len(gestures_data)} gestures...")
        
        for i, gesture in enumerate(gestures_data):
            seq = self.extract_sequence_features(gesture)
            if seq is not None:
                features.append(seq)
                labels.append(gesture['character'])
            else:
                print(f"⚠️ Skipped gesture {i} due to missing data")

        if len(features) == 0:
            raise ValueError("❌ لا توجد بيانات صالحة للتدريب.")

        X = np.array(features)
        y = np.array(labels)

        print(f"✅ Extracted {len(X)} sequences with shape {X.shape}")

        # ✅ فحص توزيع التسميات
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"📊 Label distribution: {dict(zip(unique_labels, counts))}")

        # تقسيم البيانات
        if fixed_indices:
            X_train = X[fixed_indices['train']]
            X_val = X[fixed_indices['val']]
            X_test = X[fixed_indices['test']]
            y_train = y[fixed_indices['train']]
            y_val = y[fixed_indices['val']]
            y_test = y[fixed_indices['test']]
        else:
            stratify_labels = y if len(np.unique(y)) > 1 else None
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=self.test_size,
                stratify=stratify_labels, random_state=self.random_state
            )

            val_size_adj = self.val_size / (1 - self.test_size)
            stratify_temp = y_temp if len(np.unique(y_temp)) > 1 else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adj,
                stratify=stratify_temp, random_state=self.random_state
            )

        print(f"📁 Split sizes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # ======================================================
        # ✅ Robust Scaling على بيانات التدريب فقط
        # ======================================================
        n_features = X_train.shape[2]
        
        print("🔧 Applying RobustScaler...")
        X_train_flat = X_train.reshape(-1, n_features)
        self.scaler.fit(X_train_flat)

        # تطبيق المعايرة
        X_train_scaled = self.scaler.transform(X_train_flat).reshape(X_train.shape)
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

        # ✅ فحص النتائج بعد المعايرة
        print(f"📊 After scaling - Train range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
        print(f"📊 After scaling - Mean: {X_train_scaled.mean():.2f}, Std: {X_train_scaled.std():.2f}")

        # ======================================================
        # ✅ Label Encoding محسّن
        # ======================================================
        self.label_encoder.fit(y_train)
        y_train_enc = self.label_encoder.transform(y_train)

        print(f"🎯 Label classes: {self.label_encoder.classes_}")

        def safe_encode(y_vals):
            encoded = []
            for lbl in y_vals:
                if lbl in self.label_encoder.classes_:
                    encoded.append(self.label_encoder.transform([lbl])[0])
                else:
                    # استخدام الصنف الأكثر شيوعاً كبديل
                    encoded.append(0)  # أو يمكن استخدام np.argmax(np.bincount(y_train_enc))
            return np.array(encoded)

        y_val_enc = safe_encode(y_val)
        y_test_enc = safe_encode(y_test)

        # ======================================================
        # 🧾 معلومات التقسيم مفصلة
        # ======================================================
        split_info = {
            "train_samples": len(y_train_enc),
            "val_samples": len(y_val_enc),
            "test_samples": len(y_test_enc),
            "train_distribution": dict(zip(*np.unique(y_train_enc, return_counts=True))),
            "val_distribution": dict(zip(*np.unique(y_val_enc, return_counts=True))),
            "test_distribution": dict(zip(*np.unique(y_test_enc, return_counts=True))),
            "feature_range_after_scaling": {
                "min": float(X_train_scaled.min()),
                "max": float(X_train_scaled.max()),
                "mean": float(X_train_scaled.mean()),
                "std": float(X_train_scaled.std())
            }
        }

        print("✅ Data splitting and preprocessing completed successfully!")
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train_enc, y_val_enc, y_test_enc, split_info, fixed_indices

    # ======================================================
    # 📈 حفظ إحصائيات الميزات - مُحسّن
    # ======================================================
    def plot_feature_distribution(self, gestures_data: List[Dict]):
        features_to_save = ['x', 'y', 'pressure', 'angle', 'delta_ms']
        agg = {}

        print(f"📈 Analyzing feature distribution for {len(gestures_data)} gestures...")

        for gesture in gestures_data:
            char = gesture['character']
            if char not in agg:
                agg[char] = {feat: [] for feat in features_to_save}

            frames = gesture.get('frames', [])
            if not frames and 'points' in gesture:
                frames = [{'points': gesture['points'], 'delta_ms': gesture.get('delta_ms', 16)}]

            for frame in frames:
                points = frame.get('points', [])
                delta_ms = max(frame.get('delta_ms', 16), 1)
                
                if not points:
                    agg[char]['delta_ms'].append(delta_ms)
                    continue

                for feat in features_to_save:
                    if feat == 'delta_ms':
                        agg[char]['delta_ms'].append(delta_ms)
                    else:
                        values = [p.get(feat, 0) for p in points]
                        agg[char][feat].extend(values)

        rows = []
        for char, feats in agg.items():
            row = {'character': char, 'total_samples': sum(len(v) for v in feats.values())}
            for feat, values in feats.items():
                arr = np.array(values)
                if arr.size == 0:
                    row.update({
                        f'{feat}_mean': 0.0,
                        f'{feat}_std': 0.0,
                        f'{feat}_min': 0.0,
                        f'{feat}_max': 0.0,
                        f'{feat}_nonzero_count': 0
                    })
                else:
                    non_zero = arr[arr != 0]
                    row.update({
                        f'{feat}_mean': float(np.mean(arr)),
                        f'{feat}_std': float(np.std(arr)),
                        f'{feat}_min': float(np.min(arr)),
                        f'{feat}_max': float(np.max(arr)),
                        f'{feat}_nonzero_count': len(non_zero)
                    })
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv("gesture_features_analysis.csv", index=False, encoding='utf-8-sig')
        print("✅ Gesture features analysis saved to gesture_features_analysis.csv")
        
        # عرض ملخص سريع
        print("\n📊 Feature Analysis Summary:")
        print(df[['character', 'total_samples', 'x_mean', 'y_mean', 'pressure_mean']].to_string(index=False))
        
        return df