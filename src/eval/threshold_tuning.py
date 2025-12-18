from __future__ import annotations
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

MODEL_DIR = os.path.join("src", "models", "artifacts")
DATA_DIR = os.path.join("data", "processed")
OUT_DIR = os.path.join("src", "models", "artifacts")

def load_split(name: str):
    X = pd.read_parquet(os.path.join(DATA_DIR, f"X_{name}.parquet"))
    y = pd.read_parquet(os.path.join(DATA_DIR, f"y_{name}.parquet"))["incident"].astype(int)
    return X, y

def align_cols(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in X.columns:
            X[c] = 0
    return X[cols]

def choose_threshold_max_f1(prec, rec, thr):
    # thr has length = len(prec)-1
    f1 = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thr[idx]), float(prec[idx]), float(rec[idx]), float(f1[idx])

def choose_threshold_min_recall(prec, rec, thr, target_recall=0.80):
    # Find highest precision among points meeting recall constraint
    mask = rec[:-1] >= target_recall
    if not np.any(mask):
        return None
    idx = int(np.argmax(prec[:-1][mask]))
    thr_idx = np.where(mask)[0][idx]
    return float(thr[thr_idx]), float(prec[thr_idx]), float(rec[thr_idx])

def choose_threshold_alert_rate(probs: np.ndarray, alert_rate=0.01):
    # Alert on top X% highest-risk samples
    q = 1.0 - alert_rate
    return float(np.quantile(probs, q))

def main():
    model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))

    X_val, y_val = load_split("val")
    X_val = align_cols(X_val, cols)

    probs = model.predict_proba(X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, probs)

    # 1) Max F1 threshold
    t_f1, p_f1, r_f1, f1 = choose_threshold_max_f1(prec, rec, thr)

    # 2) Recall constraint threshold
    t_r = choose_threshold_min_recall(prec, rec, thr, target_recall=0.80)

    # 3) Alert-rate threshold (top 1% by default)
    t_alert = choose_threshold_alert_rate(probs, alert_rate=0.01)

    results = {
        "max_f1": {"threshold": t_f1, "precision": p_f1, "recall": r_f1, "f1": f1},
        "recall>=0.80": None if t_r is None else {"threshold": t_r[0], "precision": t_r[1], "recall": t_r[2]},
        "top_1pct_alerts": {"threshold": t_alert},
        "notes": "Choose based on ops tolerance: recall-first for safety, alert-rate to control noise."
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot PR curve
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Validation)")
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "pr_curve_val.png"), dpi=160, bbox_inches="tight")

    print("Saved:")
    print(f"- {os.path.join(OUT_DIR, 'thresholds.json')}")
    print(f"- {os.path.join(OUT_DIR, 'pr_curve_val.png')}")
    print("\nThreshold recommendations:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
