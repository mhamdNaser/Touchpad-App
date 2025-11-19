# scripts/inspect_model.py
import torch
import joblib

def inspect_trained_model():
    print("🔍 فحص النموذج المدرب...")
    
    # تحميل state_dict لمعرفة البنية الحقيقية
    state_dict = torch.load('artifacts/encoder.pth', map_location='cpu')
    
    print("📋 مفاتيح state_dict:")
    for key in state_dict.keys():
        print(f"  - {key}: {state_dict[key].shape}")
    
    # تحميل KMeans و Mapping للتحقق
    try:
        kmeans = joblib.load('artifacts/gesture_kmeans.joblib')
        print(f"✅ KMeans: {kmeans.n_clusters} clusters")
    except Exception as e:
        print(f"❌ KMeans: {e}")
    
    try:
        mapping = joblib.load('artifacts/gesture_mapping.joblib')
        print(f"✅ Mapping: {len(mapping)} clusters -> letters")
        print(f"   Sample: {dict(list(mapping.items())[:5])}")
    except Exception as e:
        print(f"❌ Mapping: {e}")

if __name__ == '__main__':
    inspect_trained_model()