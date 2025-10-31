def recognize_gesture(data):
    # هنا تربط مع الموديل المدرب (مثلاً PyTorch أو TensorFlow)
    # مثال تجريبي فقط:
    if len(data) > 10:
        return "أ"
    else:
        return "ب"
