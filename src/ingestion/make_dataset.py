from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class DatasetConfig:
    n_rows: int = 25000
    incident_rate: float = 0.03  # rare failures
    seed: int = 42


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def main() -> None:
    cfg = DatasetConfig()
    rng = np.random.default_rng(cfg.seed)

    # Telemetry-like features
    cpu_p95 = rng.normal(55, 15, cfg.n_rows).clip(1, 100)
    mem_p95 = rng.normal(60, 18, cfg.n_rows).clip(1, 100)
    req_rate = rng.lognormal(mean=5.2, sigma=0.35, size=cfg.n_rows)
    err_5xx = rng.poisson(lam=0.5, size=cfg.n_rows)
    latency_p95 = rng.normal(180, 55, cfg.n_rows).clip(20, 2000)

    deploy_recent = rng.integers(0, 2, size=cfg.n_rows)
    region = rng.choice(
        ["blr", "mum", "sin", "fra"],
        size=cfg.n_rows,
        p=[0.45, 0.25, 0.2, 0.1],
    )
    service = rng.choice(
        ["payments", "auth", "search", "orders", "profile"],
        size=cfg.n_rows,
    )

    # Risk score (explainable)
    risk = (
        0.035 * (cpu_p95 - 50)
        + 0.03 * (mem_p95 - 55)
        + 0.002 * (latency_p95 - 150)
        + 0.55 * deploy_recent
        + 0.28 * (err_5xx > 2).astype(float)
        + 0.00015 * (req_rate - np.median(req_rate))
    )

    # ---- CALIBRATION (fix class imbalance) ----
    target = cfg.incident_rate
    lo = np.min(risk) - 50
    hi = np.max(risk) + 50

    for _ in range(40):
        mid = (lo + hi) / 2
        p_mid = sigmoid(risk - mid).mean()
        if p_mid > target:
            lo = mid
        else:
            hi = mid

    b = (lo + hi) / 2
    p = sigmoid(risk - b)
    incident = (rng.random(cfg.n_rows) < p).astype(int)

    df = pd.DataFrame(
        {
            "cpu_p95": cpu_p95,
            "mem_p95": mem_p95,
            "req_rate": req_rate,
            "err_5xx": err_5xx,
            "latency_p95": latency_p95,
            "deploy_recent": deploy_recent,
            "region": region,
            "service": service,
            "incident": incident,
        }
    )

    out_dir = os.path.join("data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "telemetry_incidents.csv")
    df.to_csv(out_path, index=False)

    print(
        f"Wrote: {out_path} | rows={len(df)} | incident_rate={df['incident'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
