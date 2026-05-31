"""Per-trial features derived from the long-format dataframe.

Adds:
    outcome        : float  bank[t+1] - bank[t]; the per-trial reward signal for RL.
                            Last trial of each (subject, condition) gets NaN.
    prev_choice    : int    choice on the immediately prior trial within condition (NaN at trial 1)
    prev_outcome   : float  outcome on the immediately prior trial within condition (NaN at trial 1)
    rolling_rate_ss: float  exponentially-smoothed local reward rate for SS choices
    rolling_rate_ll: float  exponentially-smoothed local reward rate for LL choices
    dist_to_death  : float  bank - loss_rate * time_left  (projected end-of-condition bank if no further responding)
    budget_sign    : str    "positive" or "negative" from Table 2
    loss_rate      : float  programmed loss rate for the condition (pt/s)
    net_rate       : float  programmed net rate (pt/s if SS-only)

State discretizers for the state-action Q-table models are also provided here.
"""

import numpy as np
import pandas as pd

from . import config


def add_outcomes(df):
    """Compute per-trial outcome = bank delta within (subject, condition)."""
    df = df.sort_values(["subject", "condition", "trial"]).reset_index(drop=True)
    df["outcome"] = df.groupby(["subject", "condition"])["bank"].diff().shift(-1)
    return df


def add_previous_trial(df):
    """prev_choice and prev_outcome within (subject, condition)."""
    df = df.sort_values(["subject", "condition", "trial"]).reset_index(drop=True)
    grp = df.groupby(["subject", "condition"])
    df["prev_choice"] = grp["choice"].shift(1)
    df["prev_outcome"] = grp["outcome"].shift(1)
    return df


def add_local_rates(df, alpha=0.2):
    """Exponentially smoothed local reward rate per option per (subject, condition).

    On each trial, only the chosen option's rate updates (consistent with melioration).
    """
    df = df.sort_values(["subject", "condition", "trial"]).reset_index(drop=True)
    df["rolling_rate_ss"] = np.nan
    df["rolling_rate_ll"] = np.nan

    for (sub, cond), idx in df.groupby(["subject", "condition"]).groups.items():
        idx = list(idx)
        r_ss, r_ll = 0.0, 0.0
        for i in idx:
            ch = df.at[i, "choice"]
            out = df.at[i, "outcome"]
            df.at[i, "rolling_rate_ss"] = r_ss
            df.at[i, "rolling_rate_ll"] = r_ll
            if not np.isnan(out):
                if ch == 1:
                    r_ss = (1 - alpha) * r_ss + alpha * out
                else:
                    r_ll = (1 - alpha) * r_ll + alpha * out
    return df


def add_budget_features(df):
    """Add condition metadata and distance-to-death (projected end-of-condition bank).

    dist_to_death = bank - loss_rate * time_left
    If you stopped responding right now, this is your projected end-bank.
    Positive => you survive without responding.
    Negative => you must earn (loss_rate*time_left - bank) more points to survive.
    """
    df = df.copy()
    df["loss_rate"] = df["condition"].map(lambda c: config.CONDITIONS[c]["loss_rate"])
    df["net_rate"] = df["condition"].map(lambda c: config.CONDITIONS[c]["net_rate"])
    df["budget_sign"] = df["condition"].map(lambda c: config.CONDITIONS[c]["sign"])
    df["dist_to_death"] = df["bank"] - df["loss_rate"] * df["time_left"]
    return df


NORMALIZATION_MODES = (
    "per_subject_max",       # divide by max |outcome| within subject (default)
    "per_subject_z",         # z-score within subject
    "sample_z",              # z-score across the full sample
    "per_condition_z",       # z-score within (subject, condition)
    "none",                  # raw outcomes
)


def add_outcome_normalized(df, mode="per_subject_max"):
    """Compute outcome_norm under one of the registered normalization modes.

    Modes:
      per_subject_max  : divide by max |outcome| within subject.
      per_subject_z    : (outcome - subject_mean) / subject_sd.
      sample_z         : (outcome - sample_mean) / sample_sd.
      per_condition_z  : (outcome - cell_mean) / cell_sd within (subject, condition).
      none             : raw outcomes (outcome_norm = outcome).
    """
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f"Unknown normalization mode: {mode!r}")
    df = df.copy()
    if mode == "per_subject_max":
        df["outcome_norm"] = df.groupby("subject")["outcome"].transform(
            lambda s: s / max(abs(s.dropna()).max(), 1e-6)
        )
    elif mode == "per_subject_z":
        df["outcome_norm"] = df.groupby("subject")["outcome"].transform(
            lambda s: (s - s.dropna().mean()) / max(s.dropna().std(ddof=0), 1e-6)
        )
    elif mode == "sample_z":
        valid = df["outcome"].dropna()
        mu = valid.mean()
        sd = max(valid.std(ddof=0), 1e-6)
        df["outcome_norm"] = (df["outcome"] - mu) / sd
    elif mode == "per_condition_z":
        df["outcome_norm"] = df.groupby(["subject", "condition"])["outcome"].transform(
            lambda s: (s - s.dropna().mean()) / max(s.dropna().std(ddof=0), 1e-6)
        )
    elif mode == "none":
        df["outcome_norm"] = df["outcome"]
    return df


def build_features(df, normalization_mode="per_subject_max"):
    """Run the full feature pipeline."""
    df = add_outcomes(df)
    df = add_previous_trial(df)
    df = add_local_rates(df)
    df = add_budget_features(df)
    df = add_outcome_normalized(df, mode=normalization_mode)
    return df


# ---- State discretizers for state-action Q-tables ----

def _bin(x, edges):
    """Bin a value into integer index using edges. NaN returns -1."""
    if pd.isna(x):
        return -1
    return int(np.digitize([x], edges)[0])


# Discretization edges chosen for compact state space.
DELAY_EDGES = [3, 5, 9, 17]            # bins: <3, 3-5, 5-9, 9-17, >17 (5 bins)
BANK_EDGES = [5, 15, 30, 60]           # bins: <5, 5-15, 15-30, 30-60, >60 (5 bins)
TIME_EDGES = [50, 150, 250]            # bins: <50, 50-150, 150-250, >250 (4 bins)


def state_operant(row):
    """State for the operant-features Q-table.

    Tuple of (delay_bin, prev_choice, prev_outcome_sign).
    Uses prev_choice (0/1/-1 for NaN) and sign of prev_outcome (+1/0/-1).
    """
    delay_bin = _bin(row["delay_ll"], DELAY_EDGES)
    prev_ch = int(row["prev_choice"]) if not pd.isna(row["prev_choice"]) else -1
    if pd.isna(row["prev_outcome"]):
        prev_out_sign = -2  # sentinel for missing
    else:
        prev_out_sign = int(np.sign(row["prev_outcome"]))
    return (delay_bin, prev_ch, prev_out_sign)


def state_operant_budget(row):
    """State for the operant + budget-features Q-table.

    Adds binned bank and time_left to the operant state.
    """
    base = state_operant(row)
    bank_bin = _bin(row["bank"], BANK_EDGES)
    time_bin = _bin(row["time_left"], TIME_EDGES)
    return base + (bank_bin, time_bin)


if __name__ == "__main__":
    df = pd.read_parquet(config.TRANSFORMED_DIR / "exp1_long.parquet")
    df = build_features(df)
    print(df.head(15).to_string())
    print()
    print("Outcome stats by condition:")
    print(df.groupby("condition")["outcome"].describe()[["count", "mean", "std", "min", "max"]])
    print()
    print("Sample states (P29 Chicken first 5):")
    sample = df[(df["subject"] == "Exp1_10_P29") & (df["condition"] == "Chicken")].head(5)
    for _, r in sample.iterrows():
        print(f"  trial {int(r['trial'])}: state_op={state_operant(r)}, state_opb={state_operant_budget(r)}")

    out = config.TRANSFORMED_DIR / "exp1_features.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved features to {out}")
