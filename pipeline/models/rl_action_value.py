"""Action-value RL models: Q-learning variants.

Adapted from BehavioralDynamics. State is implicit (action-value only, no state
features). Five variants:
    - q_learning      : basic Q-learning with one α, softmax β
    - q_dual_alpha    : separate learning rates for positive vs. negative RPE
    - q_forgetting    : unchosen option decays toward 0.5
    - q_dynamic_alpha : surprise-modulated learning rate
    - q_condition_aware : separate α per condition, shared β

Data schema: sdf has columns `choice` (1=SS, 0=LL), `outcome`, `condition`.
Q[0] = SS value, Q[1] = LL value. Choice encoding flipped to match BehavioralDynamics
convention where "1" means action A.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .._modelutil import logistic, bernoulli_nll_acc


def fit_rl_action_value(sdf: pd.DataFrame, n_starts: int = 10) -> list[dict]:
    """Fit all action-value RL variants on one subject. Pooled across conditions
    plus a condition-aware variant that gets separate α per condition."""
    # Convert to BehavioralDynamics convention: choice_a=1 means SS
    choices = sdf["choice"].values.astype(int)  # 1=SS, 0=LL
    # Use normalized outcomes (per-subject) so β / α live in a sensible range
    outcomes = sdf["outcome_norm"].values.astype(float)
    # Replace NaN outcomes (last trial of each condition) with 0 — they contribute
    # to choice likelihood but not to the RPE update (we skip those updates).
    valid_outcome = ~np.isnan(outcomes)
    outcomes_filled = np.where(valid_outcome, outcomes, 0.0)
    n_obs = len(choices)
    rows = []

    if n_obs < 2:
        return rows

    rows.append(_row("q_learning",       _fit_qlearning(choices, outcomes_filled, valid_outcome, n_starts), n_obs))
    rows.append(_row("q_dual_alpha",     _fit_dual_alpha(choices, outcomes_filled, valid_outcome, n_starts), n_obs))
    rows.append(_row("q_forgetting",     _fit_forgetting(choices, outcomes_filled, valid_outcome, n_starts), n_obs))
    rows.append(_row("q_dynamic_alpha",  _fit_dynamic_alpha(choices, outcomes_filled, valid_outcome, n_starts), n_obs))

    # Condition-aware: requires condition column
    conds = sdf["condition"].values
    rows.append(_row("q_condition_aware",
                     _fit_condition_aware(choices, outcomes_filled, valid_outcome, conds, n_starts),
                     n_obs))

    return [r for r in rows if r is not None]


def _row(model, fit_result, n_obs):
    if fit_result is None:
        return None
    nll = fit_result["nll"]
    n_params = fit_result["n_params"]
    aic = 2 * n_params + 2 * nll
    bic = n_params * np.log(max(n_obs, 1)) + 2 * nll
    denom = n_obs - n_params - 1
    aicc = aic + 2 * n_params * (n_params + 1) / denom if denom > 0 else np.nan
    row = {
        "model_family": "rl_action_value",
        "model": model,
        "n_params": n_params,
        "n_obs": n_obs,
        "nll": nll,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "accuracy": fit_result.get("accuracy", np.nan),
    }
    for k, v in fit_result.get("params", {}).items():
        row[f"param_{k}"] = v
    return row


# ---- Core Q-learning variants ----

def q_learning_pss(choices, rewards, valid_out, alpha, beta):
    """Per-trial P(choose SS) for basic Q-learning. Single source of the model
    dynamics, used by both the fitter (via _q_loop_nll) and predictions.py.
    Q[0]=LL, Q[1]=SS to match choice coding."""
    Q = np.array([0.0, 0.0])
    p = np.empty(len(choices))
    for t in range(len(choices)):
        p[t] = logistic(beta * (Q[1] - Q[0]))
        if valid_out[t]:
            Q[choices[t]] += alpha * (rewards[t] - Q[choices[t]])
    return p


def _q_loop_nll(choices, rewards, valid_out, alpha, beta):
    return bernoulli_nll_acc(
        choices, q_learning_pss(choices, rewards, valid_out, alpha, beta))


def _fit_qlearning(choices, rewards, valid_out, n_starts):
    def f(p):
        a = _sigmoid(p[0]); b = np.exp(p[1])
        nll, _ = _q_loop_nll(choices, rewards, valid_out, a, b)
        return nll
    best = _multi_start(f, n_starts, [(-5, 5), (-2, 5)])
    a = _sigmoid(best.x[0]); b = np.exp(best.x[1])
    _, acc = _q_loop_nll(choices, rewards, valid_out, a, b)
    return {"nll": best.fun, "n_params": 2, "accuracy": acc,
            "params": {"alpha": a, "beta": b}}


def dual_alpha_pss(choices, rewards, valid_out, ap, an, beta):
    Q = np.array([0.0, 0.0])
    p = np.empty(len(choices))
    for t in range(len(choices)):
        p[t] = logistic(beta * (Q[1] - Q[0]))
        if valid_out[t]:
            rpe = rewards[t] - Q[choices[t]]
            alpha = ap if rpe > 0 else an
            Q[choices[t]] += alpha * rpe
    return p


def _dual_alpha_nll(choices, rewards, valid_out, ap, an, beta):
    return bernoulli_nll_acc(
        choices, dual_alpha_pss(choices, rewards, valid_out, ap, an, beta))


def _fit_dual_alpha(choices, rewards, valid_out, n_starts):
    def f(p):
        ap = _sigmoid(p[0]); an = _sigmoid(p[1]); b = np.exp(p[2])
        nll, _ = _dual_alpha_nll(choices, rewards, valid_out, ap, an, b)
        return nll
    best = _multi_start(f, n_starts, [(-5, 5), (-5, 5), (-2, 5)])
    ap = _sigmoid(best.x[0]); an = _sigmoid(best.x[1]); b = np.exp(best.x[2])
    _, acc = _dual_alpha_nll(choices, rewards, valid_out, ap, an, b)
    return {"nll": best.fun, "n_params": 3, "accuracy": acc,
            "params": {"alpha_pos": ap, "alpha_neg": an, "beta": b}}


def forgetting_pss(choices, rewards, valid_out, alpha, beta, forget):
    Q = np.array([0.0, 0.0])
    p = np.empty(len(choices))
    for t in range(len(choices)):
        p[t] = logistic(beta * (Q[1] - Q[0]))
        if valid_out[t]:
            chosen = choices[t]; unchosen = 1 - chosen
            rpe = rewards[t] - Q[chosen]
            Q[chosen] += alpha * rpe
            Q[unchosen] += forget * (0.0 - Q[unchosen])  # decay toward 0 (neutral)
    return p


def _forgetting_nll(choices, rewards, valid_out, alpha, beta, forget):
    return bernoulli_nll_acc(
        choices, forgetting_pss(choices, rewards, valid_out, alpha, beta, forget))


def _fit_forgetting(choices, rewards, valid_out, n_starts):
    def f(p):
        a = _sigmoid(p[0]); b = np.exp(p[1]); g = _sigmoid(p[2])
        nll, _ = _forgetting_nll(choices, rewards, valid_out, a, b, g)
        return nll
    best = _multi_start(f, n_starts, [(-5, 5), (-2, 5), (-5, 5)])
    a = _sigmoid(best.x[0]); b = np.exp(best.x[1]); g = _sigmoid(best.x[2])
    _, acc = _forgetting_nll(choices, rewards, valid_out, a, b, g)
    return {"nll": best.fun, "n_params": 3, "accuracy": acc,
            "params": {"alpha": a, "beta": b, "forget": g}}


def dynamic_alpha_pss(choices, rewards, valid_out, a_base, a_gain, beta, decay):
    Q = np.array([0.0, 0.0])
    prev_un_rpe = 0.0
    p = np.empty(len(choices))
    for t in range(len(choices)):
        p[t] = logistic(beta * (Q[1] - Q[0]))
        if valid_out[t]:
            rpe = rewards[t] - Q[choices[t]]
            alpha_t = min(1.0, a_base + a_gain * prev_un_rpe)
            Q[choices[t]] += alpha_t * rpe
            prev_un_rpe = decay * prev_un_rpe + (1 - decay) * abs(rpe)
    return p


def _dynamic_alpha_nll(choices, rewards, valid_out, a_base, a_gain, beta, decay):
    return bernoulli_nll_acc(choices, dynamic_alpha_pss(
        choices, rewards, valid_out, a_base, a_gain, beta, decay))


def _fit_dynamic_alpha(choices, rewards, valid_out, n_starts):
    def f(p):
        ab = _sigmoid(p[0]); ag = _sigmoid(p[1]); b = np.exp(p[2]); d = _sigmoid(p[3])
        nll, _ = _dynamic_alpha_nll(choices, rewards, valid_out, ab, ag, b, d)
        return nll
    best = _multi_start(f, n_starts, [(-5, 5), (-5, 5), (-2, 5), (-5, 5)])
    ab = _sigmoid(best.x[0]); ag = _sigmoid(best.x[1])
    b = np.exp(best.x[2]); d = _sigmoid(best.x[3])
    _, acc = _dynamic_alpha_nll(choices, rewards, valid_out, ab, ag, b, d)
    return {"nll": best.fun, "n_params": 4, "accuracy": acc,
            "params": {"alpha_base": ab, "alpha_gain": ag, "beta": b, "decay": d}}


def condition_aware_pss(choices, rewards, valid_out, conds, alphas_by_cond, beta):
    Q = np.array([0.0, 0.0])
    p = np.empty(len(choices))
    for t in range(len(choices)):
        p[t] = logistic(beta * (Q[1] - Q[0]))
        if valid_out[t]:
            alpha = alphas_by_cond[conds[t]]
            rpe = rewards[t] - Q[choices[t]]
            Q[choices[t]] += alpha * rpe
    return p


def _condition_aware_nll(choices, rewards, valid_out, conds, alphas_by_cond, beta):
    return bernoulli_nll_acc(choices, condition_aware_pss(
        choices, rewards, valid_out, conds, alphas_by_cond, beta))


def _fit_condition_aware(choices, rewards, valid_out, conds, n_starts):
    unique_conds = sorted(set(conds))
    n_alphas = len(unique_conds)
    cond_to_idx = {c: i for i, c in enumerate(unique_conds)}

    def f(p):
        alphas = {c: _sigmoid(p[cond_to_idx[c]]) for c in unique_conds}
        beta = np.exp(p[n_alphas])
        nll, _ = _condition_aware_nll(choices, rewards, valid_out, conds, alphas, beta)
        return nll

    bounds = [(-5, 5)] * n_alphas + [(-2, 5)]
    best = _multi_start(f, n_starts, bounds)
    alphas = {c: float(_sigmoid(best.x[cond_to_idx[c]])) for c in unique_conds}
    beta = float(np.exp(best.x[n_alphas]))
    _, acc = _condition_aware_nll(choices, rewards, valid_out, conds, alphas, beta)
    params = {f"alpha_{c}": a for c, a in alphas.items()}
    params["beta"] = beta
    return {"nll": best.fun, "n_params": n_alphas + 1, "accuracy": acc, "params": params}


# ---- Helpers ----

def _sigmoid(x):
    """Parameter transform (unconstrained -> (0,1)) used in the fit objectives."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _multi_start(func, n_starts, bounds, seed=42):
    best = None
    rng = np.random.default_rng(seed)
    for _ in range(n_starts):
        x0 = rng.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        try:
            res = minimize(func, x0, method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue
    if best is None:
        x0 = np.zeros(len(bounds))
        best = minimize(func, x0, method="L-BFGS-B", bounds=bounds)
    return best
