"""Generate per-trial predicted P(SS) sequences for every (subject, model).

For each fitted model in fits_by_subject.parquet, replay the model through the
subject's choice sequence using the fitted parameters and emit per-trial
predicted probability of choosing the smaller-sooner option.

Output: results/predictions.parquet with columns
    subject, group, condition, trial, scope, model_family, model,
    observed_choice, predicted_p_ss
"""

import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config
from .feature_engineering import state_operant, state_operant_budget
from .models import rl_action_value as rl_av
from .models import rl_state_action as rl_sa
from .models import historical_dynamics as hist

warnings.simplefilter("ignore")


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ---- Per-model prediction functions ----

def predict_random(sdf, params):
    return np.full(len(sdf), 0.5)


def predict_energy_budget(sdf, params):
    beta = params.get("param_beta", 0.0)
    scale = params.get("param_scale", 1.0)
    dist = sdf["dist_to_death"].values.astype(float)
    return _logistic(beta * (dist / max(scale, 1e-6)))


def predict_energy_budget_threshold(sdf, params):
    beta = params.get("param_beta", 0.0)
    tau = params.get("param_tau", 0.0)
    scale = params.get("param_scale", 1.0)
    dist = sdf["dist_to_death"].values.astype(float)
    return _logistic(beta * ((dist / max(scale, 1e-6)) - tau))


def predict_energy_budget_zscore(sdf, params):
    """Recompute z-score difference and apply softmax with fitted beta."""
    from .models.energy_budget import _compute_zscore_diff
    beta = params.get("param_beta", 0.0)
    z_diff = _compute_zscore_diff(sdf)
    return _logistic(beta * z_diff)


def predict_energy_budget_categorical(sdf, params):
    """Use per-condition base rates from the fitted row to predict trial-level P(SS)."""
    p_ll_pos = params.get("param_p_ll_pos", 0.5)
    p_ll_neg = params.get("param_p_ll_neg", 0.5)
    sign = sdf["budget_sign"].values
    p_ll = np.where(sign == "positive", p_ll_pos, p_ll_neg)
    return 1.0 - p_ll


def predict_mvt(sdf, params):
    """MVT with rho fixed to the condition loss rate."""
    from .models.energy_budget import _mvt_components
    beta = params.get("param_beta", 0.0)
    scale = params.get("param_scale", 1.0)
    A, B = _mvt_components(sdf)
    rho = sdf["loss_rate"].values.astype(float)
    d = (A + rho * B) / max(scale, 1e-6)
    return _logistic(beta * d)


def predict_mvt_fitted_rho(sdf, params):
    """MVT with rho taken from the fitted per-subject parameter."""
    from .models.energy_budget import _mvt_components
    beta = params.get("param_beta", 0.0)
    rho = params.get("param_rho", 0.0)
    scale = params.get("param_scale", 1.0)
    A, B = _mvt_components(sdf)
    d = (A + rho * B) / max(scale, 1e-6)
    return _logistic(beta * d)


def predict_bias(sdf, params):
    return np.full(len(sdf), params.get("param_p", 0.5))


def predict_wsls(sdf, params):
    choices = sdf["choice"].values.astype(int)
    outcomes = sdf["outcome_norm"].values
    p_sw = params.get("param_p_stay_win", 0.7)
    p_sl = params.get("param_p_shift_lose", 0.3)
    p_ss = np.full(len(choices), 0.5)
    for t in range(1, len(choices)):
        prev_ch = choices[t - 1]
        prev_won = (not np.isnan(outcomes[t - 1])) and outcomes[t - 1] > 0
        p_stay = p_sw if prev_won else (1 - p_sl)
        p_ss[t] = p_stay if prev_ch == 1 else (1 - p_stay)
    return p_ss


def predict_logistic(sdf, params):
    # Re-fit on the fly since logistic is cheap. Uses same features as baselines.
    from sklearn.linear_model import LogisticRegression
    X = sdf[["prev_choice", "prev_outcome"]].copy()
    X["choice_x_outcome"] = X["prev_choice"] * X["prev_outcome"]
    y = sdf["choice"].values.astype(int)
    valid = X.notna().all(axis=1)
    p_ss = np.full(len(sdf), 0.5)
    if valid.sum() >= 2 and len(np.unique(y[valid.values])) >= 2:
        model = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
        model.fit(X[valid].values, y[valid.values])
        p_ss[valid.values] = model.predict_proba(X[valid].values)[:, 1]
    return p_ss


# Historical dynamics. Per-trial dynamics live in
# pipeline.models.historical_dynamics (single source of truth).
def predict_melioration(sdf, params):
    choices = sdf["choice"].values.astype(int)
    rewards = sdf["outcome_norm"].fillna(0).values
    return hist.melioration_pss(choices, rewards,
                                params.get("param_alpha", 0.1),
                                params.get("param_beta", 1.0))


def predict_kinetic(sdf, params):
    choices = sdf["choice"].values.astype(int)
    local_rates = hist._get_local_rates(sdf, config.ROLLING_WINDOW)
    return hist.kinetic_pss(choices, local_rates,
                            params.get("param_k", 0.01),
                            params.get("param_beta", 1.0))


def predict_behavioral_momentum(sdf, params):
    choices = sdf["choice"].values.astype(int)
    rewards = sdf["outcome_norm"].fillna(0).values
    conds = sdf["condition"].values
    return hist.momentum_pss(choices, rewards, conds,
                             params.get("param_c", 0.1),
                             params.get("param_d", 0.01),
                             params.get("param_beta", 1.0))


def predict_hill_climbing(sdf, params):
    choices = sdf["choice"].values.astype(int)
    rewards = sdf["outcome_norm"].fillna(0).values
    # ICI from time_left (per subject x condition; last trial of each block -> 1.0)
    sub = sdf.reset_index(drop=True)
    diffs = sub.groupby(["subject", "condition"])["time_left"].diff(periods=-1).abs().values
    ici = np.where(np.isnan(diffs), 1.0, diffs)
    return hist.hill_climbing_pss(choices, rewards, ici,
                                  params.get("param_A_recency", 1.0),
                                  params.get("param_beta", 1.0))


def predict_ratio_invariance(sdf, params):
    choices = sdf["choice"].values.astype(int)
    rewards = sdf["outcome_norm"].fillna(0).values
    return hist.ratio_invariance_pss(choices, rewards,
                                     params.get("param_omega", 0.0),
                                     params.get("param_beta", 1.0))


# RL action-value. The per-trial dynamics live in pipeline.models.rl_action_value
# (single source of truth); these wrappers just unpack fitted params and replay.
def predict_q_learning(sdf, params):
    choices = sdf["choice"].values.astype(int)
    rewards = sdf["outcome_norm"].fillna(0).values
    valid = ~np.isnan(sdf["outcome_norm"].values)
    return rl_av.q_learning_pss(choices, rewards, valid,
                                params.get("param_alpha", 0.1),
                                params.get("param_beta", 1.0))


def predict_q_dual_alpha(sdf, params):
    choices = sdf["choice"].values.astype(int)
    rewards = sdf["outcome_norm"].fillna(0).values
    valid = ~np.isnan(sdf["outcome_norm"].values)
    return rl_av.dual_alpha_pss(choices, rewards, valid,
                                params.get("param_alpha_pos", 0.1),
                                params.get("param_alpha_neg", 0.1),
                                params.get("param_beta", 1.0))


def predict_q_forgetting(sdf, params):
    choices = sdf["choice"].values.astype(int)
    rewards = sdf["outcome_norm"].fillna(0).values
    valid = ~np.isnan(sdf["outcome_norm"].values)
    return rl_av.forgetting_pss(choices, rewards, valid,
                                params.get("param_alpha", 0.1),
                                params.get("param_beta", 1.0),
                                params.get("param_forget", 0.0))


def predict_q_dynamic_alpha(sdf, params):
    choices = sdf["choice"].values.astype(int)
    rewards = sdf["outcome_norm"].fillna(0).values
    valid = ~np.isnan(sdf["outcome_norm"].values)
    return rl_av.dynamic_alpha_pss(choices, rewards, valid,
                                   params.get("param_alpha_base", 0.1),
                                   params.get("param_alpha_gain", 0.1),
                                   params.get("param_beta", 1.0),
                                   params.get("param_decay", 0.5))


def predict_q_condition_aware(sdf, params):
    choices = sdf["choice"].values.astype(int)
    rewards = sdf["outcome_norm"].fillna(0).values
    valid = ~np.isnan(sdf["outcome_norm"].values)
    conds = sdf["condition"].values
    alpha_by = {c: params.get(f"param_alpha_{c}", 0.1)
                for c in ["Chicken", "Crab", "Turtle", "Piranha"]}
    return rl_av.condition_aware_pss(choices, rewards, valid, conds, alpha_by,
                                     params.get("param_beta", 1.0))


# State-action Q-table
def predict_q_table_operant(sdf, params):
    return _predict_q_table(sdf, params, state_operant)


def predict_q_table_operant_budget(sdf, params):
    return _predict_q_table(sdf, params, state_operant_budget)


def _predict_q_table(sdf, params, state_fn):
    # Per-trial dynamics live in pipeline.models.rl_state_action (single source).
    sdf = sdf.reset_index(drop=True)
    states = [state_fn(row) for _, row in sdf.iterrows()]
    choices = sdf["choice"].values.astype(int)
    rewards = sdf["outcome_norm"].fillna(0).values
    valid = ~np.isnan(sdf["outcome_norm"].values)
    return rl_sa.q_state_action_pss(states, choices, rewards, valid,
                                    params.get("param_alpha", 0.1),
                                    params.get("param_beta", 1.0))


def predict_human_as_qtable_operant(sdf, params):
    return _predict_human_qtable(sdf, state_operant)


def predict_human_as_qtable_operant_budget(sdf, params):
    return _predict_human_qtable(sdf, state_operant_budget)


def _predict_human_qtable(sdf, state_fn):
    sdf = sdf.reset_index(drop=True)
    states = [state_fn(row) for _, row in sdf.iterrows()]
    choices = sdf["choice"].values.astype(int)
    Q = {}
    p = np.zeros(len(choices))
    for t, s in enumerate(states):
        q = Q.get(s, np.array([0.0, 0.0]))
        if q[0] == q[1]:
            # Use majority class so far as predicted probability
            p[t] = float(np.mean(choices[:t])) if t > 0 else 0.5
        else:
            total = q.sum()
            p[t] = q[1] / total  # P(SS)
        q = Q.setdefault(s, np.array([0.0, 0.0]))
        q[choices[t]] += 1.0
    return p


# ---- Dispatcher ----

PREDICT_FNS = {
    "random": predict_random,
    "bias": predict_bias,
    "wsls": predict_wsls,
    "logistic": predict_logistic,
    "energy_budget": predict_energy_budget,
    "energy_budget_threshold": predict_energy_budget_threshold,
    "energy_budget_zscore": predict_energy_budget_zscore,
    "energy_budget_categorical": predict_energy_budget_categorical,
    "mvt": predict_mvt,
    "mvt_fitted_rho": predict_mvt_fitted_rho,
    "melioration": predict_melioration,
    "kinetic": predict_kinetic,
    "behavioral_momentum": predict_behavioral_momentum,
    "hill_climbing": predict_hill_climbing,
    "ratio_invariance": predict_ratio_invariance,
    "q_learning": predict_q_learning,
    "q_dual_alpha": predict_q_dual_alpha,
    "q_forgetting": predict_q_forgetting,
    "q_dynamic_alpha": predict_q_dynamic_alpha,
    "q_condition_aware": predict_q_condition_aware,
    "q_table_operant": predict_q_table_operant,
    "q_table_operant_budget": predict_q_table_operant_budget,
    "human_as_qtable_operant": predict_human_as_qtable_operant,
    "human_as_qtable_operant_budget": predict_human_as_qtable_operant_budget,
}


def predict_one_subject(subject, sdf, fits_subject, scope="all"):
    """For one subject and scope, generate per-trial predictions for every model."""
    rows = []
    for _, fit in fits_subject[fits_subject["scope"] == scope].iterrows():
        model = fit["model"]
        if model not in PREDICT_FNS:
            continue
        # Filter sdf to the appropriate condition if scope=='condition'
        if scope == "condition":
            cdf = sdf[sdf["condition"] == fit["condition"]].copy()
        else:
            cdf = sdf
        if len(cdf) == 0:
            continue
        try:
            p_ss = PREDICT_FNS[model](cdf, fit.to_dict())
        except Exception:
            p_ss = np.full(len(cdf), 0.5)
        for i, (_, r) in enumerate(cdf.iterrows()):
            rows.append({
                "subject": subject,
                "group": fit["group"],
                "condition": r["condition"],
                "trial": r["trial"],
                "scope": scope,
                "model_family": fit["model_family"],
                "model": model,
                "observed_choice": r["choice"],
                "predicted_p_ss": float(p_ss[i]),
            })
    return rows


def run_predictions(features_df, fits_df, n_jobs=None, verbose=True):
    n_jobs = n_jobs or config.N_JOBS
    subjects = features_df["subject"].unique()
    if verbose:
        print(f"Generating predictions for {len(subjects)} subjects across {len(PREDICT_FNS)} models")

    def _do(sub):
        sdf = features_df[features_df["subject"] == sub].sort_values(
            ["condition", "trial"]).reset_index(drop=True)
        fits_sub = fits_df[fits_df["subject"] == sub]
        rows = []
        rows.extend(predict_one_subject(sub, sdf, fits_sub, scope="all"))
        rows.extend(predict_one_subject(sub, sdf, fits_sub, scope="condition"))
        return rows

    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(_do)(sub) for sub in subjects
    )
    rows = [r for sub_rows in results for r in sub_rows]
    df = pd.DataFrame(rows)
    return df


def main():
    features = pd.read_parquet(config.TRANSFORMED_DIR / "exp1_features.parquet")
    fits = pd.read_parquet(config.RESULTS_DIR / "fits_by_subject.parquet")
    preds = run_predictions(features, fits, verbose=True)
    out = config.RESULTS_DIR / "predictions.parquet"
    preds.to_parquet(out, index=False)
    print(f"\nSaved {len(preds)} prediction rows to {out}")


if __name__ == "__main__":
    main()
