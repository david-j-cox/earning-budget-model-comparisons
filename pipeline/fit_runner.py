"""Per-subject fitting orchestrator.

For each subject, fits every model from every family on two scopes:
  - scope="all"      : pooled across all four conditions for that subject
  - scope=<condname> : on the subject's data within one condition only

The latter is the per-condition secondary breakdown. Results are concatenated
into a single long-format dataframe.
"""

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config
from .models import (baselines, historical_dynamics, rl_action_value,
                     rl_state_action, energy_budget)

warnings.simplefilter("ignore")


MODEL_FAMILIES = [
    ("baseline", baselines.fit_baselines),
    ("energy_budget", energy_budget.fit_energy_budget),
    ("historical_dynamics", historical_dynamics.fit_historical),
    ("rl_action_value", rl_action_value.fit_rl_action_value),
    ("rl_state_action", rl_state_action.fit_state_action_rl),
]


def _safe_fit(fit_fn, sdf, **kwargs):
    """Run a model family's fit function with errors swallowed (so one bad subject
    doesn't kill the whole run)."""
    try:
        return fit_fn(sdf, **kwargs)
    except Exception as e:
        return [{"_error": f"{fit_fn.__module__}: {type(e).__name__}: {e}"}]


def fit_one_subject(subject, sdf, n_starts=10):
    """Fit all models for one subject under both scopes."""
    rows = []

    # Scope 1: pooled across conditions
    for family_name, fit_fn in MODEL_FAMILIES:
        results = _safe_fit(fit_fn, sdf, n_starts=n_starts) if family_name != "baseline" \
                  else _safe_fit(fit_fn, sdf)
        for r in results:
            if "_error" in r:
                continue
            r["subject"] = subject
            r["group"] = sdf["group"].iloc[0]
            r["scope"] = "all"
            r["condition"] = "all"
            rows.append(r)

    # Scope 2: per condition. No trial-count threshold — death is data.
    for cond, cdf in sdf.groupby("condition"):
        if len(cdf) < 2:
            # Cannot define a per-trial prediction with fewer than 2 trials
            continue
        for family_name, fit_fn in MODEL_FAMILIES:
            results = _safe_fit(fit_fn, cdf, n_starts=n_starts) if family_name != "baseline" \
                      else _safe_fit(fit_fn, cdf)
            for r in results:
                if "_error" in r:
                    continue
                r["subject"] = subject
                r["group"] = sdf["group"].iloc[0]
                r["scope"] = "condition"
                r["condition"] = cond
                rows.append(r)

    return rows


def run_all(features_df, n_starts=None, n_jobs=None, verbose=True):
    """Run fits for every subject. Returns a single long-format dataframe."""
    n_starts = n_starts or config.N_STARTS
    n_jobs = n_jobs or config.N_JOBS

    subjects = features_df["subject"].unique()
    if verbose:
        print(f"Fitting {len(subjects)} subjects across {len(MODEL_FAMILIES)} families "
              f"using {n_jobs} workers, n_starts={n_starts}")

    def _do_one(sub):
        sdf = features_df[features_df["subject"] == sub].sort_values(
            ["condition", "trial"]).reset_index(drop=True)
        return fit_one_subject(sub, sdf, n_starts=n_starts)

    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(_do_one)(sub) for sub in subjects
    )

    rows = [r for subj_rows in results for r in subj_rows]
    df = pd.DataFrame(rows)

    # Order key columns first
    front = ["subject", "group", "scope", "condition", "model_family", "model",
             "n_params", "n_obs", "nll", "aic", "bic", "accuracy"]
    other = [c for c in df.columns if c not in front]
    df = df[front + other]
    return df
