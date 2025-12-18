from __future__ import annotations
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_DIR = os.path.join("data", "processed")
MODEL_DIR = os.path.join("src", "models", "artifacts")

def load_split(name: str):
    X = pd.read_parquet(os.path.join(DATA_DIR, f"X_{name}.parquet"))
    y = pd.read_parquet(os.path.join(DATA_DIR, f"y_{name}.parquet"))["incident"].astype(int)
    return X, y

def main() -> None:
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Baseline: Logistic Regression (good for explainability)
    lr = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=1))
    ])
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_val)[:, 1]
    lr_ap = average_precision_score(y_val, lr_probs)
    print("\n[Baseline] LogisticRegression AP:", round(lr_ap, 4))

    # Stronger: Random Forest with class_weight
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_val)[:, 1]
    rf_ap = average_precision_score(y_val, rf_probs)
    print("[Model] RandomForest AP:", round(rf_ap, 4))

    best = rf if rf_ap >= lr_ap else lr
    best_name = "random_forest" if best is rf else "log_reg"

    # Save model + feature columns for inference consistency
    joblib.dump(best, os.path.join(MODEL_DIR, "model.joblib"))
    joblib.dump(list(X_train.columns), os.path.join(MODEL_DIR, "feature_columns.joblib"))

    print(f"\nSaved best model: {best_name} -> src/models/artifacts/")
    print("\nClassification report (val):")
    preds = (rf_probs >= 0.5).astype(int) if best is rf else (lr_probs >= 0.5).astype(int)
    print(classification_report(y_val, preds, digits=4))

if __name__ == "__main__":
    main()
