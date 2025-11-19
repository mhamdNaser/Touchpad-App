#!/usr/bin/env python3
"""
النص الرئيسي لتدريب نموذج التجميع
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.gesture_data_loader import GestureDataLoader
from app.services.gesture_cluster_trainer import GestureClusterTrainer

def main():
    print("🚀 بدء تدريب نموذج التجميع للإيماءات...")
    
    # 1. تحميل البيانات
    print("📥 جاري تحميل البيانات...")
    data_loader = GestureDataLoader(
        target_frames=30,      # تقليل عدد الإطارات لتقليل التعقيد
        target_points=15,      # تقليل عدد النقاط
        rotate_normalize=True,
        center_to_zero=True
    )
    
    processed_gestures = data_loader.load_all_gestures()
    
    if not processed_gestures:
        print("❌ فشل في تحميل البيانات")
        return
    
    print(f"✅ تم تحميل {len(processed_gestures)} إيماءة معالجة")
    
    # 2. تدريب النموذج
    print("🎯 بدء تدريب نموذج التجميع...")
    trainer = GestureClusterTrainer()
    
    # استخدام auto-detect للعثور على العدد الأمثل للكتل
    cluster_labels = trainer.train(processed_gestures, auto_detect_k=True)
    
    # 3. حفظ النموذج
    trainer.save_model("models/gesture_cluster_model.pkl")
    
    # 4. عرض النتائج
    print("\n" + "="*50)
    print("🎊 تم الانتهاء من التدريب بنجاح!")
    print("="*50)
    
    # اختبار التنبؤ على بعض العينات
    print("\n🧪 اختبار التنبؤ على عينات عشوائية:")
    import random
    test_samples = random.sample(processed_gestures, min(5, len(processed_gestures)))
    
    for i, sample in enumerate(test_samples):
        true_char = sample.get("character", "unknown")
        predicted_char = trainer.predict(sample)
        
        status = "✅" if predicted_char == true_char else "❌"
        print(f"   {status} العينة {i+1}: الحقيقي='{true_char}', المتوقع='{predicted_char}'")

if __name__ == "__main__":
    main()