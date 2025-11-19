import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "ADVANCED_features.csv"

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_PATH)
if "character" not in df.columns:
    raise ValueError("❌ ملف الفيتشر يجب أن يحتوي عمود 'character'.")

y = df["character"]

# ============================================================
# CLEAN NON-NUMERIC OR NON-FEATURE COLUMNS
# ============================================================
invalid_cols = [c for c in df.columns if c == "character" or not np.issubdtype(df[c].dtype, np.number)]
X = df.drop(columns=invalid_cols)

# ============================================================
# FEATURE IMPORTANCE USING RANDOM FOREST
# ============================================================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

model = RandomForestClassifier(n_estimators=250, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X, y_encoded)

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_,
    "mean": X.mean().values,
    "std": X.std().values,
    "min": X.min().values,
    "max": X.max().values
}).sort_values(by="importance", ascending=False)

# ============================================================
# SAVE SHORT REPORT
# ============================================================
importance_df.to_csv("feature_short_report.csv", index=False)

# ============================================================
# PRINT TOP FEATURES SUMMARY
# ============================================================
top_features = importance_df.head(20)
print("🌟 Top 20 Features by Importance:")
print(top_features[["feature", "importance"]])

# OPTIONAL: list features to consider dropping
low_importance = importance_df[importance_df["importance"] < 0.001]
print(f"\n⚠️ Features with very low importance (could drop): {len(low_importance)}")
print(low_importance["feature"].tolist())

print("\n✅ Short feature report saved to 'feature_short_report.csv'")
