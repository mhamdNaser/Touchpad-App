# app/services/gesture_utils.py
import numpy as np
import pandas as pd

def distance(p1, p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))

def path_length(points):
    points = np.array(points)
    return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

def summarize_processed(gestures_data, processed_gestures):
    """
    ملخص للبيانات بعد المعالجة
    gestures_data: list of gesture dict (للحرف)
    processed_gestures: np.array [num_gestures, resample_frames, 2]
    """
    print(f"\n📊 Summarizing processed gestures")
    num_gestures = len(processed_gestures)
    print(f"✅ Total gestures: {num_gestures}\n")

    # توزيع الأحرف
    characters = [g['character'] for g in gestures_data]
    print("🅰️ Character distribution:")
    pd.Series(characters).value_counts().sort_index().plot(kind='bar', figsize=(12,4), title='Character Distribution')
    print(pd.Series(characters).value_counts(), "\n")

    # عدد الفريمات لكل إيماءة
    frames_per_gesture = [g.shape[0] for g in processed_gestures]
    print(f"🖐 Frames per gesture (after resample):")
    print(f"Min: {np.min(frames_per_gesture)}, Max: {np.max(frames_per_gesture)}, Mean: {np.mean(frames_per_gesture):.2f}\n")

    # إحصاءات مختصرة عن x و y
    x_vals = processed_gestures[:,:,0].flatten()
    y_vals = processed_gestures[:,:,1].flatten()
    print("📌 Feature statistics (x, y) after normalization & resample:")
    stats_df = pd.DataFrame({'x': x_vals, 'y': y_vals}).describe().transpose()
    print(stats_df, "\n")

def summarize_csv(csv_path):
    print(f"\n📊 Summarizing CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Total gestures: {len(df)}\n")

    # عدد كل حرف
    print("🅰️ Character distribution:")
    print(df['character'].value_counts(), "\n")

    # عدد الفريمات لكل إيماءة
    num_points_cols = [c for c in df.columns if c.startswith('x')]
    df['num_frames'] = df[num_points_cols].notna().sum(axis=1)
    print(f"🖐 Frames per gesture:")
    print(f"Min: {df['num_frames'].min()}, Max: {df['num_frames'].max()}, Mean: {df['num_frames'].mean():.2f}\n")

    # إحصاءات مختصرة عن x, y, pressure
    print("📌 Feature statistics (x, y, pressure):")
    features_cols = [c for c in df.columns if any(c.startswith(p) for p in ['x','y','pressure'])]
    print(df[features_cols].describe().transpose(), "\n")



