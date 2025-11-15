import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ============================================================
#                CONFIG
# ============================================================
DATA_PATH = "ADVANCED_features.csv"
OUTPUT_DIR = "feature_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
#                LOAD DATA
# ============================================================
print("📥 Loading features...")
df = pd.read_csv(DATA_PATH)

if "character" not in df.columns:
    raise ValueError("❌ ملف الفيتشر يجب أن يحتوي عمود 'character'.")

y = df["character"]

# ============================================================
#         REMOVE NON-NUMERIC OR NON-FEATURE COLUMNS
# ============================================================
invalid_cols = []

for col in df.columns:
    if col == "character":
        continue
    if not np.issubdtype(df[col].dtype, np.number):
        invalid_cols.append(col)
        continue
    # ID columns should be removed
    if "id" in col.lower():
        invalid_cols.append(col)
    if "gesture" in col.lower():
        invalid_cols.append(col)
    if col.lower() in ["orig_frame_count"]:
        invalid_cols.append(col)

df_clean = df.drop(columns=invalid_cols)

print(f"⚠️ Ignored non-feature columns: {invalid_cols}")
print(f"🔢 Final usable features: {df_clean.shape[1]-1}")

X = df_clean.drop(columns=["character"])

# ============================================================
#         SAFE BOX PLOTS (numeric cols only)
# ============================================================
print("📊 Generating Box Plots safely...")

for feature in X.columns:
    try:
        plt.figure(figsize=(7, 3))
        sns.boxplot(y=df_clean[feature])
        plt.title(f"Box Plot - {feature}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/box_{feature}.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ Skipped {feature} (boxplot error: {e})")

print("✅ Box plots saved.")

# ============================================================
#           HISTOGRAMS
# ============================================================
print("📊 Generating Histograms...")

for feature in X.columns:
    try:
        plt.figure(figsize=(7, 3))
        plt.hist(df_clean[feature], bins=40)
        plt.title(f"Histogram - {feature}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/hist_{feature}.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ Skipped {feature} (hist error: {e})")

print("✅ Histograms saved.")

# ============================================================
#       CORRELATION HEATMAP (up to 200 features)
# ============================================================
print("🔥 Generating Correlation Heatmap...")

corr_features = X.iloc[:, : min(200, len(X.columns))]

plt.figure(figsize=(14, 12))
sns.heatmap(corr_features.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
plt.close()

print("✅ Heatmap saved.")

# ============================================================
#     FEATURE IMPORTANCE USING RANDOM FOREST
# ============================================================
print("🌲 Calculating Feature Importance...")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

model = RandomForestClassifier(
    n_estimators=250,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y_encoded)

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

importance_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)

plt.figure(figsize=(12, 10))
sns.barplot(
    x="importance",
    y="feature",
    data=importance_df.iloc[:40]
)
plt.title("Top 40 Feature Importance")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png")
plt.close()

print("🎯 Feature importance saved.")
print("✨ Analysis completed! Check the folder:", OUTPUT_DIR)
