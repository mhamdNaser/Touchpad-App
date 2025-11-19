# debug_data_quality.py
import numpy as np
import pickle
from app.services.gesture_data_loader import GestureDataLoader
from app.services.advanced_feature_extractor import AdvancedFeatureExtractor

def analyze_data_quality():
    """تحليل جودة البيانات الأصلية"""
    print("🔍 DEEP DATA QUALITY ANALYSIS...")
    
    # تحميل البيانات
    data_loader = GestureDataLoader(api_url="https://api.sydev.site/api/gestures")
    gestures_data = data_loader.load_all_gestures()
    
    # تحليل كل حرف على حدة
    char_analysis = {}
    
    for gesture in gestures_data:
        char = gesture['character']
        if char not in char_analysis:
            char_analysis[char] = {
                'count': 0,
                'total_frames': 0,
                'total_points': 0,
                'has_frames': 0,
                'has_points': 0
            }
        
        char_analysis[char]['count'] += 1
        
        frames = gesture.get('frames', [])
        if frames:
            char_analysis[char]['has_frames'] += 1
            char_analysis[char]['total_frames'] += len(frames)
            
            for frame in frames:
                points = frame.get('points', [])
                char_analysis[char]['total_points'] += len(points)
        else:
            # البيانات القديمة
            points = gesture.get('points', [])
            if points:
                char_analysis[char]['has_points'] += 1
                char_analysis[char]['total_points'] += len(points)
    
    print("\n📊 CHARACTER DATA QUALITY ANALYSIS:")
    print("="*50)
    for char, stats in char_analysis.items():
        avg_frames = stats['total_frames'] / stats['count'] if stats['count'] > 0 else 0
        avg_points = stats['total_points'] / stats['count'] if stats['count'] > 0 else 0
        frame_ratio = stats['has_frames'] / stats['count'] if stats['count'] > 0 else 0
        point_ratio = stats['has_points'] / stats['count'] if stats['count'] > 0 else 0
        
        print(f"🔤 {char}:")
        print(f"   Samples: {stats['count']}")
        print(f"   Has frames: {frame_ratio:.1%}")
        print(f"   Has points: {point_ratio:.1%}")
        print(f"   Avg frames: {avg_frames:.1f}")
        print(f"   Avg points: {avg_points:.1f}")
        
        # تحذير إذا كانت البيانات ضعيفة
        if avg_points < 10 or stats['count'] < 100:
            print(f"   ⚠️  LOW QUALITY DATA!")

if __name__ == "__main__":
    analyze_data_quality()