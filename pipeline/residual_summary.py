"""Per-family trial-level residual summary (Results, "Model Fits").

Reproduces the per-family median absolute trial-level residual reported in the
Results section (e.g., baseline .068/.093, Human-as-Q-Table .045/.065,
historical behavior-dynamic .158/.131). These numbers were previously computed
ad hoc; this module commits the exact methodology so they are reproducible.

Methodology (verified to reproduce the unchanged families to three decimals):
  - scope == "all" predictions only.
  - For each (group, condition, trial) cell, take the mean observed P(SS) across
    subjects (deduplicated on subject; cell kept only if >= MIN_COUNT subjects)
    and, per model, the mean predicted P(SS) across subjects (same threshold).
  - Residual for a (model, condition, trial) cell = |obs_mean - pred_mean|.
  - A family's value (per group) is the median of these residuals pooled over
    every model in the family and every condition x trial cell. No trial cap.

Output: results/residual_summary.csv  (one row per family; columns for each group)
"""

import numpy as np
import pandas as pd

from . import config
from .make_figures import FAMILY_MODELS, COND_ORDER

MIN_COUNT = 5


def family_residual(preds, models, group, scope="all", min_count=MIN_COUNT):
    """Pooled median |observed_mean - predicted_mean| over (model, condition,
    trial) cells for one family in one group."""
    d = preds[(preds["scope"] == scope) & (preds["group"] == group)]
    vals = []
    for cond in COND_ORDER:
        sub = d[d["condition"] == cond]
        obs = (sub.drop_duplicates(subset=["subject", "trial"])
               .groupby("trial")["observed_choice"].agg(["mean", "count"]))
        obs = obs[obs["count"] >= min_count]["mean"]
        for m in models:
            md = (sub[sub["model"] == m]
                  .groupby("trial")["predicted_p_ss"].agg(["mean", "count"]))
            md = md[md["count"] >= min_count]["mean"]
            j = obs.index.intersection(md.index)
            vals.extend(np.abs(obs.loc[j].values - md.loc[j].values))
    return float(np.median(vals)) if vals else np.nan


def main():
    preds = pd.read_parquet(config.RESULTS_DIR / "predictions.parquet")
    available = set(preds["model"].unique())
    rows = []
    for fam, models in FAMILY_MODELS.items():
        present = [m for m in models if m in available]
        if not present:
            continue
        rows.append({
            "family": fam,
            "n_models": len(present),
            "median_abs_residual_grp05": family_residual(preds, present, "05"),
            "median_abs_residual_grp10": family_residual(preds, present, "10"),
        })
    df = pd.DataFrame(rows).sort_values("median_abs_residual_grp05")
    out = config.RESULTS_DIR / "residual_summary.csv"
    df.to_csv(out, index=False)
    print(f"  Wrote {out}")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
