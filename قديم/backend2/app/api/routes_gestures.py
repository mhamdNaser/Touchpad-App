# app/api/routes_gestures.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

from app.services.prediction_pipeline import ClusteringPredictionPipeline

router = APIRouter()
logger = logging.getLogger(__name__)

# محاولة تحميل النموذج
try:
    predictor = ClusteringPredictionPipeline()
    logger.info("🎯 تم تهيئة pipeline التنبؤ")
    if predictor.is_ready:
        logger.info("✅ النموذج جاهز للاستخدام")
    else:
        logger.warning("⚠️ النموذج غير جاهز - تأكد من وجود ملفات artifacts")
except Exception as e:
    logger.warning(f"⚠️ فشل في تحميل النموذج: {e}")
    predictor = None

class Point(BaseModel):
    x: float
    y: float
    pressure: float = 1.0

class Frame(BaseModel):
    ts: int
    frame_id: int
    points: List[Point]
    delta_ms: Optional[int] = 16

class GesturePayload(BaseModel):
    start_time: int
    end_time: int
    duration_ms: int
    frame_count: int
    frames: List[Frame]

@router.post("/predict")
def predict(gesture: GesturePayload):
    if predictor is None or not predictor.is_ready:
        raise HTTPException(
            status_code=503, 
            detail="النموذج غير جاهز للتنبؤ. يرجى التدريب أولاً أو التحقق من ملفات النموذج في مجلد artifacts."
        )
    
    try:
        logger.info(f"📨 استقبال إيماءة: {len(gesture.frames)} إطار، {gesture.duration_ms} مللي ثانية")
        
        # تحويل إلى تنسيق القاموس مع الحفاظ على الهيكل الأصلي
        gesture_dict = {
            "frames": [
                {
                    "ts": frame.ts,
                    "timestamp": frame.ts,  # نضيف timestamp كمفتاح بديل
                    "delta_ms": frame.delta_ms or 16,
                    "points": [
                        {
                            "x": point.x,
                            "y": point.y,
                            "pressure": point.pressure
                        }
                        for point in frame.points
                    ]
                }
                for frame in gesture.frames
            ],
            "duration_ms": gesture.duration_ms,
            "start_time": gesture.start_time,
            "end_time": gesture.end_time,
            "frame_count": gesture.frame_count
        }
        
        result = predictor.predict_gesture(gesture_dict)
        
        # تسجيل النتيجة
        if result["success"]:
            logger.info(f"✅ تنبؤ ناجح: {result['predicted_letter']}")
        else:
            logger.error(f"❌ فشل في التنبؤ: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ خطأ في معالجة الطلب: {e}")
        raise HTTPException(status_code=400, detail=f"خطأ في معالجة الإيماءة: {str(e)}")

@router.get("/health")
def health_check():
    if predictor:
        status = predictor.get_status()
        return {
            "status": "ready" if status["is_ready"] else "not_ready",
            "model_loaded": status["is_ready"],
            "message": "النموذج جاهز للتنبؤ" if status["is_ready"] else "النموذج غير محمل",
            "details": status
        }
    else:
        return {
            "status": "not_ready",
            "model_loaded": False,
            "message": "لم يتم تهيئة النموذج"
        }

@router.get("/model-status")
def model_status():
    """تفاصيل حالة النموذج"""
    if predictor:
        return predictor.get_status()
    else:
        return {
            "is_ready": False,
            "model_loaded": False,
            "message": "النموذج غير محمل"
        }

@router.get("/debug-sample")
def debug_sample():
    """عينة تجريبية لاختبار المعالجة"""
    if not predictor or not predictor.is_ready:
        return {"error": "النموذج غير جاهز"}
    
    # إنشاء عينة تجريبية مشابهة للواجهة الأمامية
    sample_data = {
        "frames": [
            {
                "ts": 1763549402312,
                "delta_ms": 16,
                "points": [
                    {"x": 100, "y": 100, "pressure": 1.0},
                    {"x": 110, "y": 105, "pressure": 1.0},
                    {"x": 120, "y": 110, "pressure": 1.0}
                ]
            },
            {
                "ts": 1763549402328,
                "delta_ms": 16,
                "points": [
                    {"x": 105, "y": 102, "pressure": 1.0},
                    {"x": 115, "y": 107, "pressure": 1.0},
                    {"x": 125, "y": 112, "pressure": 1.0}
                ]
            }
        ],
        "duration_ms": 100,
        "start_time": 1763549402312,
        "end_time": 1763549402412
    }
    
    try:
        result = predictor.predict_gesture(sample_data)
        return {
            "sample_data": sample_data,
            "prediction_result": result
        }
    except Exception as e:
        return {"error": str(e)}