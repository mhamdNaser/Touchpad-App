#!/usr/bin/env python3
"""
اختبار استخراج الميزات المحسنة - نسخة تصحيح الأخطاء
"""

import sys
import os
import requests
import json

# إضافة المسار للوحدات
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.gesture_data_loader import GestureDataLoader
from app.services.advanced_feature_extractor import AdvancedFeatureExtractor

def test_api_connection():
    """اختبار اتصال API مباشرة"""
    print("🔍 اختبار اتصال API...")
    try:
        url = "https://api.sydev.site/api/gestures?page=1&per_page=10"
        response = requests.get(url, timeout=30)
        print(f"📡 حالة الرد: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            gestures = data.get("data", [])
            print(f"✅ تم جلب {len(gestures)} إيماءة من API")
            
            if gestures:
                print("📋 عينة من البيانات:")
                gesture = gestures[0]
                print(f"   - ID: {gesture.get('id')}")
                print(f"   - الحرف: {gesture.get('character')}")
                print(f"   - عدد الإطارات: {len(gesture.get('frames', []))}")
                
                # فحص أول إطار
                frames = gesture.get('frames', [])
                if frames:
                    first_frame = frames[0]
                    print(f"   - نقاط الإطار الأول: {len(first_frame.get('points', []))}")
                    if first_frame.get('points'):
                        first_point = first_frame['points'][0]
                        print(f"   - نقطة أولى: {first_point}")
                
        else:
            print(f"❌ خطأ في API: {response.status_code}")
            print(f"   النص: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ خطأ في الاتصال: {e}")

def test_data_loader():
    """اختبار محمل البيانات"""
    print("\n🔍 اختبار محمل البيانات...")
    try:
        loader = GestureDataLoader(per_page=10)  # عدد أقل للاختبار
        
        # اختبار التحميل المباشر
        page = 1
        url = f"https://api.sydev.site/api/gestures?page={page}&per_page=10"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            raw_gestures = data.get("data", [])
            print(f"📥 تم تحميل {len(raw_gestures)} إيماءة خام")
            
            if raw_gestures:
                # معالجة أول إيماءة فقط للاختبار
                raw_gesture = raw_gestures[0]
                print(f"🎯 معالجة إيماءة ID: {raw_gesture.get('id')}")
                
                # معالجة يدوية لمحاكاة _process_gesture
                frames_raw = raw_gesture.get("frames", [])
                print(f"   - الإطارات الخام: {len(frames_raw)}")
                
                if frames_raw:
                    # ترتيب الإطارات
                    frames = sorted(frames_raw, key=lambda x: x.get("timestamp") or x.get("ts") or x.get("frame_id") or 0)
                    print(f"   - الإطارات بعد الترتيب: {len(frames)}")
                    
                    # تصفية الإطارات الفارغة
                    frames = [f for f in frames if f.get("points")]
                    print(f"   - الإطارات بعد التصفية: {len(frames)}")
                    
                    if frames:
                        # تطبيع أول إطار
                        points = frames[0].get("points", [])
                        print(f"   - نقاط الإطار الأول: {len(points)}")
                        
                        if points:
                            print("✅ البيانات موجودة ويجب أن تعمل!")
                            return True
                
                print("❌ لا توجد بيانات كافية في الإيماءة")
                
        return False
        
    except Exception as e:
        print(f"❌ خطأ في محمل البيانات: {e}")
        return False

def simple_feature_test():
    """اختبار مبسط لاستخراج الميزات"""
    print("\n🔍 اختبار استخراج الميزات المبسط...")
    
    try:
        # إنشاء بيانات اختبار افتراضية
        test_gesture = {
            "id": "test_1",
            "character": "آ",
            "frames": [
                {
                    "timestamp": 0,
                    "delta_ms": 16,
                    "points": [
                        {"x": 0.0, "y": 0.0, "pressure": 1.0},
                        {"x": 1.0, "y": 1.0, "pressure": 1.0},
                        {"x": 2.0, "y": 0.5, "pressure": 1.0}
                    ]
                },
                {
                    "timestamp": 16,
                    "delta_ms": 16,
                    "points": [
                        {"x": 0.5, "y": 0.5, "pressure": 1.0},
                        {"x": 1.5, "y": 1.5, "pressure": 1.0},
                        {"x": 2.5, "y": 1.0, "pressure": 1.0}
                    ]
                }
            ]
        }
        
        extractor = AdvancedFeatureExtractor()
        features = extractor.gesture_to_feature_vector(test_gesture)
        
        print(f"✅ تم استخراج {len(features)} ميزة")
        print(f"📊 الميزات: {features}")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في استخراج الميزات: {e}")
        return False

if __name__ == "__main__":
    print("🚀 بدء اختبار استخراج الميزات - نسخة تصحيح الأخطاء...")
    
    # 1. اختبار الاتصال بالAPI
    test_api_connection()
    
    # 2. اختبار محمل البيانات
    if test_data_loader():
        # 3. إذا البيانات شغالة، جرب التحميل الكامل
        try:
            loader = GestureDataLoader(per_page=20)
            gestures = loader.load_all_gestures()
            
            if gestures:
                print(f"🎉 تم تحميل {len(gestures)} إيماءة معالجة")
                
                # اختبار استخراج الميزات
                extractor = AdvancedFeatureExtractor()
                extractor.analyze_features_by_character(gestures)
                extractor.save_gestures_to_csv(gestures, "improved_features.csv")
                
            else:
                print("❌ لم يتم تحميل أي إيماءات معالجة")
                
        except Exception as e:
            print(f"❌ خطأ في التحميل الكامل: {e}")
    
    # 4. اختبار باستخدام بيانات افتراضية
    print("\n🔍 اختبار بالبيانات الافتراضية...")
    simple_feature_test()
    
    print("\n📝 ملخص الاستكشاف:")
    print("1. إذا فشل اختبار API: المشكلة في الاتصال أو الخادم")
    print("2. إذا نجح اختبار API لكن فشل محمل البيانات: المشكلة في معالجة البيانات")
    print("3. إذا نجح الاختبار الافتراضي: الكود شغال لكن البيانات فيها مشكلة")