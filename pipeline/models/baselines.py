"""Baseline choice models: random, bias, WSLS, logistic.

Adapted from BehavioralDynamics/measuring-behavior-trajectories.
Data schema mapping:
    session_id     -> subject
    choice_a       -> choice    (1 = SS/immediate, 0 = LL/delayed)
    reward_outcome -> outcome   (bank delta per trial)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score


def fit_baselines(sdf: pd.DataFrame) -> list[dict]:
    """Fit all baseline models on one subject's data (pooled across conditions).

    sdf must have columns: choice, outcome, condition (for condition dummies).
    Returns list of dicts ready for the results table.
    """
    choices = sdf["choice"].values.astype(float)
    outcomes = sdf["outcome_norm"].values  # may contain NaN at end of conditions
    n_obs = len(choices)
    rows = []

    # Random baseline: P=0.5 for every trial
    nll_random = -np.sum(np.log(0.5) * np.ones(n_obs))
    rows.append(_row("random", 0, nll_random, n_obs, accuracy=0.5))

    # Bias: constant P(SS)
    bias = _fit_bias(choices)
    pred_acc = float(((bias["p"] >= 0.5) == (choices == 1)).mean())
    rows.append(_row("bias", 1, bias["nll"], n_obs, accuracy=pred_acc, p=bias["p"]))

    # Win-stay/lose-shift
    wsls = _fit_wsls(choices, outcomes)
    rows.append(_row("wsls", 2, wsls["nll"], n_obs, accuracy=wsls["accuracy"],
                     p_stay_win=wsls["p_stay_win"], p_shift_lose=wsls["p_shift_lose"]))

    # Logistic regression
    log = _fit_logistic(sdf)
    if log is not None:
        rows.append(_row("logistic", log["n_params"], log["nll"], log["n_obs"],
                         accuracy=log["accuracy"]))

    return rows


def _row(model, n_params, nll, n_obs, accuracy=np.nan, **params):
    aic = 2 * n_params + 2 * nll
    bic = n_params * np.log(max(n_obs, 1)) + 2 * nll
    aicc = _aicc(aic, n_params, n_obs)
    return {
        "model_family": "baseline",
        "model": model,
        "n_params": n_params,
        "n_obs": n_obs,
        "nll": nll,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "accuracy": accuracy,
        **{f"param_{k}": v for k, v in params.items()},
    }


def _aicc(aic, k, n):
    denom = n - k - 1
    if denom <= 0 or not np.isfinite(aic):
        return np.nan
    return aic + 2 * k * (k + 1) / denom


def _fit_bias(choices):
    p = float(np.clip(np.mean(choices), 1e-6, 1 - 1e-6))
    nll = -float(np.sum(choices * np.log(p) + (1 - choices) * np.log(1 - p)))
    return {"p": p, "nll": nll}


def _fit_wsls(choices, outcomes):
    """WSLS with two parameters: p(stay | win), p(shift | lose).
    A trial is a 'win' if outcome > 0, else a 'loss'. End-of-condition NaN outcomes
    are treated as losses (no information).
    """
    # Build prev-trial outcome indicator (1 if previous trial was a win)
    won = np.zeros_like(choices, dtype=int)
    for i in range(1, len(choices)):
        won[i] = 1 if (not np.isnan(outcomes[i - 1]) and outcomes[i - 1] > 0) else 0

    def neg_log_lik(params):
        p_stay_win, p_shift_lose = params
        p_stay_win = np.clip(p_stay_win, 1e-6, 1 - 1e-6)
        p_shift_lose = np.clip(p_shift_lose, 1e-6, 1 - 1e-6)
        ll = 0.0
        for t in range(1, len(choices)):
            p_stay = p_stay_win if won[t] == 1 else (1 - p_shift_lose)
            if choices[t] == choices[t - 1]:
                ll += np.log(max(p_stay, 1e-10))
            else:
                ll += np.log(max(1 - p_stay, 1e-10))
        return -ll

    result = minimize(neg_log_lik, [0.7, 0.3], bounds=[(0.01, 0.99), (0.01, 0.99)],
                      method="L-BFGS-B")
    p_sw, p_sl = result.x
    # Accuracy: predict stay if prev win and p_stay_win > 0.5, else shift; etc.
    preds = np.zeros_like(choices)
    preds[0] = choices[0]
    for t in range(1, len(choices)):
        p_stay = p_sw if won[t] == 1 else (1 - p_sl)
        preds[t] = choices[t - 1] if p_stay >= 0.5 else (1 - choices[t - 1])
    accuracy = float((preds == choices).mean())
    return {"p_stay_win": p_sw, "p_shift_lose": p_sl,
            "nll": float(result.fun), "accuracy": accuracy}


def _fit_logistic(sdf):
    """Logistic regression on prev_choice, prev_outcome, choice×outcome interaction."""
    X = sdf[["prev_choice", "prev_outcome"]].copy()
    X["choice_x_outcome"] = X["prev_choice"] * X["prev_outcome"]
    y = sdf["choice"].values.astype(int)

    valid = X.notna().all(axis=1)
    X = X[valid].values
    y = y[valid.values]

    if len(y) < 2 or len(np.unique(y)) < 2:
        # Logistic regression undefined with <2 trials or no variance in choice.
        # Emit a degenerate row so subject still appears in comparison.
        return {
            "n_params": 4,
            "nll": np.nan,
            "n_obs": len(y),
            "accuracy": np.nan,
        }

    model = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    nll = float(log_loss(y, probs, normalize=False))
    return {
        "n_params": 4,  # 3 features + intercept
        "nll": nll,
        "n_obs": len(y),
        "accuracy": float(accuracy_score(y, model.predict(X))),
    }
