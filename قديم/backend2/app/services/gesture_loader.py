# app/services/gesture_loader.py
import json
import os
import requests
from typing import List, Dict, Any
from tqdm import tqdm

class GestureLoader:
    def __init__(self, base_url="http://localhost:8000/api/gestures", cache_path="cache/raw_gestures.json"):
        self.base_url = base_url
        self.cache_path = cache_path
        self.session = requests.Session()

    def fetch_all(self, use_cache=True):
        if use_cache and os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data, {"cached": True, "count": len(data)}

        gestures = []
        page = 1
        while True:
            url = f"{self.base_url}?page={page}"
            resp = self.session.get(url, timeout=5)
            if resp.status_code != 200:
                break

            data = resp.json()
            items = data.get("data", [])
            if not items:
                break

            gestures.extend(items)
            if not data.get("links", {}).get("next"):
                break

            page += 1

        os.makedirs("cache", exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(gestures, f, ensure_ascii=False, indent=2)

        return gestures, {"cached": False, "count": len(gestures)}
