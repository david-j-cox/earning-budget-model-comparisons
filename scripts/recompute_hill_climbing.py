"""One-time migration: re-fit hill_climbing and regenerate only its prediction
rows in place, leaving every other model's rows untouched.

Used to propagate the hill_climbing model change (sign flip + per-component
rescaling; see pipeline/models/historical_dynamics.py) into the existing
results without re-fitting all 22 models in a different environment, which would
risk optimizer/library drift in the other models' reported numbers. A full
`python -m run` followed by the predictions step reproduces the same end state
from scratch when run in the original environment.

Run from the repo root:  python scripts/recompute_hill_climbing.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from pipeline import config, predictions as pr
from pipeline.models import historical_dynamics as hd


def main():
    feat = pd.read_parquet(config.TRANSFORMED_DIR / "exp1_features.parquet")
    fits = pd.read_parquet(config.RESULTS_DIR / "fits_by_subject.parquet")
    preds = pd.read_parquet(config.RESULTS_DIR / "predictions.parquet")

    # 1. Re-fit hill_climbing in place (update values, keep schema/row order).
    update_cols = ["nll", "aic", "aicc", "bic", "accuracy", "n_params",
                   "param_A_recency", "param_beta"]
    n_refit = 0
    for idx in fits.index[fits["model"] == "hill_climbing"]:
        sub = fits.at[idx, "subject"]
        scope = fits.at[idx, "scope"]
        cond = fits.at[idx, "condition"]
        sdf = (feat[feat["subject"] == sub]
               .sort_values(["condition", "trial"]).reset_index(drop=True))
        if scope == "condition":
            sdf = sdf[sdf["condition"] == cond].reset_index(drop=True)
        choices = sdf["choice"].values.astype(int)
        outcomes = sdf["outcome_norm"].values.astype(float)
        outcomes_filled = np.where(np.isnan(outcomes), 0.0, outcomes)
        ici_s = hd._compute_ici(sdf)
        fit = hd._fit_hill_climbing(choices, outcomes_filled, ici_s,
                                    config.N_STARTS)
        newrow = hd._row("hill_climbing", fit, len(choices))
        if newrow is None:
            continue
        for c in update_cols:
            if c in newrow:
                fits.at[idx, c] = newrow[c]
        n_refit += 1
    print(f"re-fit {n_refit} hill_climbing fit rows")
    fits.to_parquet(config.RESULTS_DIR / "fits_by_subject.parquet", index=False)

    # 2. Regenerate hill_climbing predictions from the new fitted parameters.
    new_rows = []
    for sub in feat["subject"].unique():
        sdf = (feat[feat["subject"] == sub]
               .sort_values(["condition", "trial"]).reset_index(drop=True))
        fits_sub_hc = fits[(fits["subject"] == sub) &
                           (fits["model"] == "hill_climbing")]
        if len(fits_sub_hc) == 0:
            continue
        new_rows.extend(pr.predict_one_subject(sub, sdf, fits_sub_hc, scope="all"))
        new_rows.extend(pr.predict_one_subject(sub, sdf, fits_sub_hc,
                                               scope="condition"))
    new_df = pd.DataFrame(new_rows)
    preds_other = preds[preds["model"] != "hill_climbing"]
    preds_out = pd.concat([preds_other, new_df[preds.columns]], ignore_index=True)
    assert len(preds_other) + len(new_df) == len(preds_out)
    preds_out.to_parquet(config.RESULTS_DIR / "predictions.parquet", index=False)
    print(f"regenerated {len(new_df)} hill_climbing prediction rows; "
          f"wrote predictions.parquet ({len(preds_out)} rows, "
          f"{len(preds_other)} non-hill rows unchanged)")


if __name__ == "__main__":
    main()
