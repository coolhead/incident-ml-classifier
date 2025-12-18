from __future__ import annotations
import os
import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report

MODEL_DIR = os.path.join("src", "models", "artifacts")
DATA_DIR = os.path.join("data", "processed")

def main() -> None:
    model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))

    X_test = pd.read_parquet(os.path.join(DATA_DIR, "X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(DATA_DIR, "y_test.parquet"))["incident"].astype(int)

    # Align columns (robustness)
    for c in cols:
        if c not in X_test.columns:
            X_test[c] = 0
    X_test = X_test[cols]

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("Test AP:", round(average_precision_score(y_test, probs), 4))
    print("Test ROC-AUC:", round(roc_auc_score(y_test, probs), 4))
    print("\nClassification report (test):")
    print(classification_report(y_test, preds, digits=4))

if __name__ == "__main__":
    main()
