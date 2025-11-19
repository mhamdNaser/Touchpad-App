# debug_test_model.py
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import traceback

def print_header(title):
    print("\n" + "="*10 + " " + title + " " + "="*10)

def safe_load_pickle(path, name):
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"✅ Loaded {name} from {path}")
        return obj
    except Exception as e:
        print(f"❌ Failed to load {name} from {path}: {e}")
        raise

def main():
    try:
        # ---- loads ----
        X_test = safe_load_pickle("X_test.pkl", "X_test")
        y_test = safe_load_pickle("y_test.pkl", "y_test")
        scaler = safe_load_pickle("scaler.pkl", "scaler")
        label_encoder = safe_load_pickle("label_encoder.pkl", "label_encoder")

        print_header("Shapes & types")
        print("X_test.shape:", getattr(X_test, "shape", None), "dtype:", getattr(X_test, "dtype", None))
        print("y_test.shape:", getattr(y_test, "shape", None), "dtype:", getattr(y_test, "dtype", None))
        print("Label classes:", getattr(label_encoder, "classes_", None))
        try:
            unique_y, counts = np.unique(y_test, return_counts=True)
            print("y_test distribution:", dict(zip(unique_y.astype(int).tolist(), counts.tolist())))
        except Exception as e:
            print("Could not compute y_test distribution:", e)

        # ---- scaler introspection ----
        print_header("Scaler info")
        for attr in ("mean_", "scale_", "var_", "n_features_in_"):
            if hasattr(scaler, attr):
                print(f"{attr}:", getattr(scaler, attr))
            else:
                print(f"{attr}: NOT FOUND on scaler")

        # ---- scaling ----
        n_samples, timesteps, n_features = X_test.shape
        print_header("Before scaling stats (first 5 samples)")
        print("X_test[0] min/max/mean/std:", np.nanmin(X_test[0]), np.nanmax(X_test[0]), np.nanmean(X_test[0]), np.nanstd(X_test[0]))
        all_min = np.nanmin(X_test)
        all_max = np.nanmax(X_test)
        print("Global min/max:", all_min, all_max)

        X_test_flat = X_test.reshape(-1, n_features)
        print("X_test_flat shape:", X_test_flat.shape)

        # Check for NaN/inf
        print("NaN in X_test:", np.isnan(X_test_flat).any(), "Inf in X_test:", np.isinf(X_test_flat).any())

        X_test_scaled_flat = scaler.transform(X_test_flat)
        print("✅ Scaler.transform OK. Scaled flat shape:", X_test_scaled_flat.shape)
        X_test_scaled = X_test_scaled_flat.reshape(n_samples, timesteps, n_features)

        print_header("After scaling stats (first 5 samples)")
        print("X_test_scaled[0] min/max/mean/std:", np.nanmin(X_test_scaled[0]), np.nanmax(X_test_scaled[0]), np.nanmean(X_test_scaled[0]), np.nanstd(X_test_scaled[0]))
        print("Scaled global min/max:", np.nanmin(X_test_scaled), np.nanmax(X_test_scaled))

        # ---- load model ----
        model_path_candidates = ["arabic_gesture_cnn_best.h5", "arabic_gesture_cnn_final.h5", "arabic_gesture_cnn_latest.h5"]
        model = None
        for p in model_path_candidates:
            try:
                model = load_model(p)
                print(f"✅ Loaded model from {p}")
                model_path = p
                break
            except Exception as e:
                print(f"Could not load model from {p}: {e}")
        if model is None:
            raise RuntimeError("No model file loaded. Put correct .h5 file in folder or update path.")

        print_header("Model summary")
        try:
            model.summary()
        except Exception:
            print("Could not print model.summary()")

        # ---- quick forward check ----
        print_header("Model input/output checks")
        # predict probabilities for first few samples
        preds_proba = model.predict(X_test_scaled[:10])
        print("preds_proba shape (first 10):", preds_proba.shape)
        print("example row probs (first 3 rows):")
        for i, row in enumerate(preds_proba[:3]):
            print(f" row {i} sum={row.sum():.6f} min={row.min():.6f} max={row.max():.6f} probs={row}")

        # check if all predicted classes are same
        y_pred_all = np.argmax(model.predict(X_test_scaled), axis=1)
        print("✅ Predictions done. y_pred shape:", y_pred_all.shape)
        unique_preds, pred_counts = np.unique(y_pred_all, return_counts=True)
        print("Predicted unique classes & counts:", dict(zip(unique_preds.tolist(), pred_counts.tolist())))

        # compare to y_test
        print_header("Classification report")
        print(classification_report(y_test, y_pred_all, zero_division=0))

        # confusion matrix basics
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred_all)
        print("Confusion matrix rows sum (true counts):", cm.sum(axis=1))
        print("Confusion matrix cols sum (pred counts):", cm.sum(axis=0))

        # ---- extra checks ----
        print_header("Extra checks")
        # are scaler.mean_ huge/zero/identical?
        if hasattr(scaler, "mean_"):
            m = np.array(scaler.mean_)
            print("scaler.mean_ stats:", m.min(), m.max(), m.mean())
        if hasattr(scaler, "scale_"):
            s = np.array(scaler.scale_)
            print("scaler.scale_ stats:", s.min(), s.max(), s.mean())

        # check for constant rows in scaled data
        stds = np.std(X_test_scaled.reshape(n_samples, -1), axis=1)
        print("std of each sample (first 10):", stds[:10])
        print("how many samples have std==0:", np.sum(stds == 0))

        # check label encoder mapping
        try:
            classes = list(label_encoder.classes_)
            mapping = {cls: idx for idx, cls in enumerate(classes)}
            print("Label encoder mapping (class -> index):", mapping)
        except Exception as e:
            print("Could not read label_encoder.classes_:", e)

    except Exception as e:
        print("ERROR during debug run:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
