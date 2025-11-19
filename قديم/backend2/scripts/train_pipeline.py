# scripts/train_pipeline.py
import argparse
import numpy as np
import torch
from tqdm import tqdm

from app.services.gesture_loader import GestureLoader
from app.services.preprocessing.resampling import Resampler
from app.services.preprocessing.normalization import Normalizer
from app.services.preprocessing.feature_extractor import FeatureExtractor

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--no-resample", action="store_true")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--no-features", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()

def main():
    args = parse_args()

    loader = GestureLoader()
    resampler = Resampler()
    normalizer = Normalizer()
    extractor = FeatureExtractor()

    summary = {}

    # ----------------- FETCH -----------------
    if not args.no_fetch:
        gestures, info = loader.fetch_all()
    else:
        with open("cache/raw_gestures.json","r",encoding="utf-8") as f:
            gestures = json.load(f)
        info = {"cached": True, "count": len(gestures)}

    summary["fetch"] = info

    processed = []

    for g in tqdm(gestures, desc="Processing"):
        frames = g.get("frames", [])
        if not frames:
            continue

        points = [[{"x": p["x"], "y": p["y"]} for p in frame] for frame in frames]
        pts = np.array([[[p["x"], p["y"]] for p in frame] for frame in frames])

        # ---------- RESAMPLE ----------
        if not args.no_resample:
            pts = resampler.resample_frames(pts)

        # ---------- NORMALIZE ----------
        if not args.no_normalize:
            pts = normalizer.normalize(pts)

        # ---------- FEATURES ----------
        if not args.no_features:
            pts = extractor.extract(pts)

        processed.append(pts)

    X = np.array(processed)
    summary["final_shape"] = X.shape

    print("\n===== SUMMARY =====")
    for k,v in summary.items():
        print(k, "=>", v)

    # -------------- TRAIN (اختياري) --------------
    if args.train:
        X_t = torch.tensor(X, dtype=torch.float32)
        print("Training on:", X_t.shape)
        # نموذج بسيط كمثال
        from torch import nn
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(X_t.shape[1]*X_t.shape[2]*X_t.shape[3], 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for epoch in range(args.epochs):
            loss = 0
            for i in range(0, len(X_t), args.batch_size):
                batch = X_t[i:i+args.batch_size]
                pred = model(batch)
                l = loss_fn(pred, torch.zeros_like(pred))
                opt.zero_grad()
                l.backward()
                opt.step()
                loss += l.item()

            print(f"Epoch {epoch+1}/{args.epochs} — Loss: {loss:.4f}")

        print("Training done. Saving model...")
        torch.save(model.state_dict(), "artifacts/model.pth")

if __name__ == "__main__":
    main()
