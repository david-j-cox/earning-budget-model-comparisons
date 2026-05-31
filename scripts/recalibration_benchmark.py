"""Benchmark behind the hill_climbing recalibration decision.

Compares the persisted (per-component rescaled) hill_climbing model against
fit-quality and calibration diagnostics, to document why per-component
standardization was adopted: it drops predicted-probability saturation from
~33% to ~1% and improves median AICc, at the cost of some thresholded accuracy.

Run from the repo root:  python scripts/recalibration_benchmark.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from pipeline import config


def diagnostics(preds, model, scope="all"):
    d = preds[(preds["scope"] == scope) & (preds["model"] == model)]
    ps = d["predicted_p_ss"].clip(1e-10, 1 - 1e-10).values
    y = d["observed_choice"].astype(float).values
    nll = -(y * np.log(ps) + (1 - y) * np.log(1 - ps))
    hit = (ps >= 0.5) == (y == 1)
    extreme = (ps < 0.05) | (ps > 0.95)
    return {
        "accuracy": round(float(hit.mean()), 3),
        "nll_per_trial": round(float(nll.mean()), 3),
        "frac_pred_near_bounds": round(float(extreme.mean()), 3),
        "frac_extreme_on_wrong_side": round(float((extreme & ~hit).mean()), 4),
    }


def main():
    preds = pd.read_parquet(config.RESULTS_DIR / "predictions.parquet")
    fits = pd.read_parquet(config.RESULTS_DIR / "fits_by_subject.parquet")
    hc = fits[(fits["model"] == "hill_climbing") & (fits["scope"] == "all")]
    print("Persisted hill_climbing (per-component rescaled):")
    for g in ["05", "10"]:
        s = hc[hc["group"] == g]
        print(f"  Group {g}: median AICc={s['aicc'].median():.2f}  "
              f"mean accuracy={s['accuracy'].mean():.3f}")
    print("\nCalibration diagnostics (scope=all), reference models:")
    for m in ["hill_climbing", "energy_budget_categorical", "random", "logistic"]:
        print(f"  {m:26s}", diagnostics(preds, m))


if __name__ == "__main__":
    main()
