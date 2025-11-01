from app.services.data_loader import DataLoader
from app.services.features import FeatureEngineer
from app.core.database import SessionLocal
import numpy as np
import pprint

def main():
    db = SessionLocal()
    loader = DataLoader(db)
    features_extractor = FeatureEngineer()

    # Load gestures for two characters ('ا' and 'ب')
    data = loader.load_gestures_data(["ا", "ب"], limit_per_char=10)  # Use 10 for testing

    print(f"Loaded {len(data)} gestures\n")

    # Extract features
    X, y = features_extractor.extract_features(data)

    print("🔹 Feature dimensions:")
    print(f"X shape: {X.shape}")  # (number of samples, number of features)
    print(f"y shape: {y.shape}")  # (number of samples, )

    # Check data types
    print(f"X dtype: {X.dtype}, y dtype: {y.dtype}")

    # Show the first sample to inspect values
    print("\nFirst gesture example after feature extraction:")
    pprint.pprint(X[0])
    print("Label:", y[0])

    # Additional check: verify no NaN values
    if np.isnan(X).any():
        print("⚠️ NaN values found in features!")
    else:
        print("✅ No NaN values in features.")

if __name__ == "__main__":
    main()
