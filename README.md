# Incident ML Classifier

**Applied ML + Production-Ready API**

A production-minded applied ML system that predicts and classifies **operational incidents** from telemetry-like signals (metrics, latency, errors, deployment signals).

Designed as an **early-warning system** for **rare incidents (~3% base rate)** with:

* explicit **threshold tuning**
* **model explainability**
* **FastAPI + Docker** deployment

---

## Motivation

Operational incidents are rare but high impact.
Static rules don’t adapt and often miss emerging failure patterns.

This project demonstrates:

* **Imbalanced classification** (rare-event ML)
* **Threshold tuning** to balance recall vs alert volume
* **Explainability** for trust and debugging
* **Production packaging** via API and Docker

---
![Precision–Recall Curve (Validation)](docs/pr_curve_val.png)

---

## Project Structure

```
data/
  raw/                  # generated dataset
  processed/            # train / val / test splits

src/
  ingestion/            # dataset generation
  features/             # feature engineering + splits
  models/               # training + saved artifacts
  eval/                 # threshold tuning + explainability
  api/                  # FastAPI inference service
```

---

## Quickstart (Local)

### 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Generate data + build features

```bash
python3 -m src.ingestion.make_dataset
python3 -m src.features.build_features
```

### 3) Train + evaluate

```bash
python3 -m src.models.train
python3 -m src.eval.evaluate
```

### 4) Threshold tuning + explainability

```bash
python3 -m src.eval.threshold_tuning
python3 -m src.eval.explain
```

**Artifacts generated**

* `src/models/artifacts/thresholds.json`
* `src/models/artifacts/pr_curve_val.png`
* `src/models/artifacts/feature_importance.csv`

---

## Run the API (Local)

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Health check

```bash
curl -s http://localhost:8000/health
```

### Predict

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_p95":72,
    "mem_p95":80,
    "req_rate":250,
    "err_5xx":4,
    "latency_p95":420,
    "deploy_recent":1,
    "region":"blr",
    "service":"payments"
  }'
```

---

## Threshold Strategy

This is a **rare-event classifier**.
The default `0.5` threshold is arbitrary and **not optimal**.

Thresholds are tuned using Precision–Recall curves:

* **max_f1** → balanced precision / recall
* **recall ≥ 0.80** → recall-first (avoid missed incidents)
* **top_1pct_alerts** → alert-volume control (ops-friendly)

Override threshold at runtime:

```bash
INCIDENT_THRESHOLD=0.25 uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Docker

### Build

```bash
docker build -t incident-ml-classifier:latest .
```

### Run (host 8001 → container 8000)

```bash
docker run -p 8001:8000 incident-ml-classifier:latest
curl -s http://localhost:8001/health
```

### Override threshold

```bash
docker run -p 8001:8000 \
  -e INCIDENT_THRESHOLD=0.25 \
  incident-ml-classifier:latest
```

---

## Model Explainability

Two complementary approaches are used:

* **Logistic Regression coefficients** (fast, interpretable)
* **Permutation importance** (model-agnostic validation)

Typical top drivers:

* CPU / memory pressure
* Recent deployment signal
* Latency spikes

This ensures predictions align with **domain intuition**, not just metrics.

---

## Project Impact

* Ability to design ML systems for **rare-event problems**
* Strong understanding of **precision–recall tradeoffs**, not just accuracy
* Experience shipping **explainable models**, not black boxes
* Skill in packaging ML into **production-ready services**
