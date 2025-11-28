import json
import csv
import numpy as np
import requests

class GestureDataLoader:
    def __init__(self, json_path=None, json_data=None, api_url=None):
        self.json_path = json_path
        self.api_url = api_url
        self.data = json_data or (self.load_json(json_path) if json_path else None)
        self.gestures = []

    def load_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_api_data(self):
        """تحميل كل البيانات من API متعدد الصفحات"""
        if not self.api_url:
            raise ValueError("API URL not provided")

        all_data = []
        page = 1
        while True:
            print(f"📡 Fetching page {page} ...")
            resp = requests.get(self.api_url, params={"page": page})
            resp.raise_for_status()
            page_data = resp.json().get("data", [])
            if not page_data:
                break
            all_data.extend(page_data)
            page += 1

        self.data = {"data": all_data}
        print(f"✅ Total gestures loaded from API: {len(all_data)}")
        return self.data

    def parse_data(self):
        """تحويل البيانات الخام إلى قائمة من الإيماءات"""
        if self.data is None:
            raise ValueError("No data loaded. Use load_json or load_api_data first.")

        gestures = []
        for item in self.data['data']:
            gesture = {
                'id': item['id'],
                'character': item['character'],
                'duration_ms': item['duration_ms'],
                'frame_count': item['frame_count'],
                'points': item['points']
            }
            gestures.append(gesture)
        self.gestures = gestures
        return gestures

    def save_to_csv_flat(self, csv_path):
        """حفظ كل الإيماءات في CSV بحيث كل صف يمثل إيماءة كاملة"""
        if not self.gestures:
            self.parse_data()

        max_points = max(g['frame_count'] for g in self.gestures)

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['gesture_id', 'character']
            for i in range(max_points):
                header += [f'x{i+1}', f'y{i+1}', f'pressure{i+1}']
            writer.writerow(header)

            for g in self.gestures:
                row = [g['id'], g['character']]
                for p in g['points']:
                    row += [p['x'], p['y'], p['pressure']]
                remaining = max_points - len(g['points'])
                row += [np.nan] * (remaining * 3)
                writer.writerow(row)
