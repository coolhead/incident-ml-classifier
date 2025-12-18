from __future__ import annotations
import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = os.path.join("data", "raw", "telemetry_incidents.csv")
OUT_DIR = os.path.join("data", "processed")

def main() -> None:
    df = pd.read_csv(RAW_PATH)

    # Basic cleaning (keep it interview-defensible)
    df = df.dropna()

    # One-hot encode categorical vars
    X = df.drop(columns=["incident"])
    y = df["incident"].astype(int)

    X = pd.get_dummies(X, columns=["region", "service"], drop_first=True)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    X_train.to_parquet(os.path.join(OUT_DIR, "X_train.parquet"), index=False)
    y_train.to_frame("incident").to_parquet(os.path.join(OUT_DIR, "y_train.parquet"), index=False)

    X_val.to_parquet(os.path.join(OUT_DIR, "X_val.parquet"), index=False)
    y_val.to_frame("incident").to_parquet(os.path.join(OUT_DIR, "y_val.parquet"), index=False)

    X_test.to_parquet(os.path.join(OUT_DIR, "X_test.parquet"), index=False)
    y_test.to_frame("incident").to_parquet(os.path.join(OUT_DIR, "y_test.parquet"), index=False)

    print("Saved processed splits to data/processed/")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Pos rate (train): {y_train.mean():.4f}")

if __name__ == "__main__":
    main()
