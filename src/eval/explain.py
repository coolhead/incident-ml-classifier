from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

MODEL_DIR = os.path.join("src", "models", "artifacts")
DATA_DIR = os.path.join("data", "processed")

def load_split(name: str):
    X = pd.read_parquet(os.path.join(DATA_DIR, f"X_{name}.parquet"))
    y = pd.read_parquet(os.path.join(DATA_DIR, f"y_{name}.parquet"))["incident"].astype(int)
    return X, y

def align_cols(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in X.columns:
            X[c] = 0
    return X[cols]

def main():
    model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))

    X_val, y_val = load_split("val")
    X_val = align_cols(X_val, cols)

    print("\n=== Model explainability ===")

    # 1) If LogisticRegression pipeline: show coefficients
    try:
        clf = model.named_steps["clf"]  # Pipeline case
        coef = clf.coef_[0]
        coef_df = pd.DataFrame({"feature": cols, "coef": coef})
        coef_df["abs_coef"] = coef_df["coef"].abs()
        top = coef_df.sort_values("abs_coef", ascending=False).head(15)
        print("\nTop LR coefficients (absolute):")
        print(top[["feature", "coef"]].to_string(index=False))
    except Exception:
        print("\nCoefficient view skipped (model is not LR pipeline).")

    # 2) Permutation importance using average precision proxy via predict_proba
    # We'll score with model's default .score isn't great for imbalance; use roc_auc-like approach:
    # Permutation importance uses scoring; for predict_proba models, roc_auc is fine.
    from sklearn.metrics import roc_auc_score, make_scorer
    scorer = make_scorer(roc_auc_score, needs_proba=True)

    pi = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=5,
        random_state=42,
        scoring=scorer,
        n_jobs=-1,
    )

    imp_df = pd.DataFrame({
        "feature": cols,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std,
    }).sort_values("importance_mean", ascending=False)

    out_path = os.path.join(MODEL_DIR, "feature_importance.csv")
    imp_df.to_csv(out_path, index=False)

    print("\nTop permutation importances (ROC-AUC impact):")
    print(imp_df.head(15).to_string(index=False))
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
