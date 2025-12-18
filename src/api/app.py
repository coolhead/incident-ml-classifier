from __future__ import annotations
import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_DIR = os.path.join("src", "models", "artifacts")

app = FastAPI(title="Incident ML Classifier", version="1.0.0")

model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))

# Load recommended threshold (default: max_f1)
DEFAULT_THRESHOLD = 0.5
threshold_path = os.path.join(MODEL_DIR, "thresholds.json")
if os.path.exists(threshold_path):
    with open(threshold_path, "r", encoding="utf-8") as f:
        t = json.load(f)
    DEFAULT_THRESHOLD = float(t.get("max_f1", {}).get("threshold", DEFAULT_THRESHOLD))

# Allow override (useful for ops tuning)
THRESHOLD = float(os.getenv("INCIDENT_THRESHOLD", DEFAULT_THRESHOLD))


class Telemetry(BaseModel):
    cpu_p95: float = Field(..., ge=0, le=100)
    mem_p95: float = Field(..., ge=0, le=100)
    req_rate: float = Field(..., ge=0)
    err_5xx: int = Field(..., ge=0)
    latency_p95: float = Field(..., ge=0)
    deploy_recent: int = Field(..., ge=0, le=1)
    region: str
    service: str


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=["region", "service"], drop_first=True)
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df[cols]


@app.get("/health")
def health():
    return {"status": "ok", "threshold": THRESHOLD}


@app.post("/predict")
def predict(t: Telemetry):
    df = pd.DataFrame([t.model_dump()])
    X = featurize(df)
    prob = float(model.predict_proba(X)[:, 1][0])
    pred = int(prob >= THRESHOLD)
    return {
        "incident_probability": prob,
        "threshold": THRESHOLD,
        "incident_predicted": pred,
    }
