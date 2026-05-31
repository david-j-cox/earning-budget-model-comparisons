"""Historical behavior-dynamic models of choice.

Five named models from the 1980s–1992 behavior dynamics literature:
  - Melioration (Herrnstein & Vaughan, 1980; Vaughan, 1981)
  - Kinetic model (Myerson & Miezin, 1980; Myerson & Hale, 1988)
  - Behavioral momentum (Nevin, 1983; Nevin & Shahan, 2011)
  - Hill-climbing / momentary maximizing (Hinson & Staddon, 1983)
  - Ratio invariance (Staddon, 1988)

Adapted from BehavioralDynamics/measuring-behavior-trajectories. Schema mapping:
  choice_a       -> choice (1=SS, 0=LL)
  reward_outcome -> outcome (bank delta)
  phase_id       -> condition (Chicken/Crab/Turtle/Piranha)
  ici_s          -> inter-click time (-diff of time_left)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .._modelutil import logistic, bernoulli_nll_acc


def fit_historical(sdf: pd.DataFrame, n_starts: int = 10, window: int = 10) -> list[dict]:
    """Fit all five historical models on one subject's pooled choices."""
    choices = sdf["choice"].values.astype(int)
    outcomes = sdf["outcome_norm"].values.astype(float)
    outcomes_filled = np.where(np.isnan(outcomes), 0.0, outcomes)
    n_obs = len(choices)
    rows = []

    if n_obs < 2:
        return rows

    rows.append(_row("melioration",
                     _fit_melioration(choices, outcomes_filled, n_starts), n_obs))
    rows.append(_row("kinetic",
                     _fit_kinetic(sdf, choices, n_starts, window), n_obs))
    # Phases here = conditions
    conds = sdf["condition"].values
    rows.append(_row("behavioral_momentum",
                     _fit_momentum(choices, outcomes_filled, conds, n_starts), n_obs))
    # Inter-click time
    ici_s = _compute_ici(sdf)
    rows.append(_row("hill_climbing",
                     _fit_hill_climbing(choices, outcomes_filled, ici_s, n_starts), n_obs))
    rows.append(_row("ratio_invariance",
                     _fit_ratio_invariance(choices, outcomes_filled, n_starts), n_obs))

    return [r for r in rows if r is not None]


def _row(model, fit, n_obs):
    if fit is None or not np.isfinite(fit["nll"]):
        return None
    nll = fit["nll"]
    n_params = fit["n_params"]
    aic = 2 * n_params + 2 * nll
    bic = n_params * np.log(max(n_obs, 1)) + 2 * nll
    denom = n_obs - n_params - 1
    aicc = aic + 2 * n_params * (n_params + 1) / denom if denom > 0 else np.nan
    row = {
        "model_family": "historical_dynamics",
        "model": model,
        "n_params": n_params,
        "n_obs": n_obs,
        "nll": nll,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "accuracy": fit.get("accuracy", np.nan),
    }
    for k, v in fit.get("params", {}).items():
        row[f"param_{k}"] = v
    return row


def _compute_ici(sdf):
    """Inter-click intervals in seconds, derived from time_left differences."""
    t = sdf["time_left"].values.astype(float)
    ici = np.full(len(t), np.nan)
    # Within (subject, condition), each next click happens at -diff(time_left)
    df = sdf.reset_index(drop=True)
    df["_ici"] = df.groupby(["subject", "condition"])["time_left"].diff(periods=-1).abs()
    ici = df["_ici"].values
    return ici


# ---- Melioration ----

def melioration_pss(choices, rewards, alpha, beta):
    """Per-trial P(choose SS) for melioration. Single source of the dynamics."""
    R_SS, R_LL = 0.0, 0.0
    p = np.empty(len(choices))
    for t in range(len(choices)):
        p[t] = logistic(beta * (R_SS - R_LL))
        if choices[t] == 1:
            R_SS = (1 - alpha) * R_SS + alpha * rewards[t]
        else:
            R_LL = (1 - alpha) * R_LL + alpha * rewards[t]
    return p


def _melioration_nll(choices, rewards, alpha, beta):
    return bernoulli_nll_acc(choices, melioration_pss(choices, rewards, alpha, beta))


def _fit_melioration(choices, rewards, n_starts):
    def f(p):
        a = _sigmoid(p[0]); b = np.exp(p[1])
        return _melioration_nll(choices, rewards, a, b)[0]
    best = _multi_start(f, n_starts, [(-5, 5), (-2, 5)])
    a = _sigmoid(best.x[0]); b = np.exp(best.x[1])
    _, acc = _melioration_nll(choices, rewards, a, b)
    return {"nll": best.fun, "n_params": 2, "accuracy": acc,
            "params": {"alpha": a, "beta": b}}


# ---- Kinetic ----

def _get_local_rates(sdf, window):
    """Rolling-window estimated reward rate per option."""
    choices = sdf["choice"].values.astype(int)
    rewards = sdf["outcome_norm"].fillna(0).values.astype(float)
    R_SS = np.full(len(choices), 0.0)
    R_LL = np.full(len(choices), 0.0)
    for t in range(window, len(choices)):
        win = slice(t - window, t)
        c = choices[win]; r = rewards[win]
        n_ss = c.sum(); n_ll = window - n_ss
        R_SS[t] = (c * r).sum() / max(n_ss, 1)
        R_LL[t] = ((1 - c) * r).sum() / max(n_ll, 1)
    return R_SS, R_LL


def kinetic_pss(choices, local_rates, k, beta):
    """Per-trial P(choose SS) for the kinetic model. `local_rates` is the
    (R_SS, R_LL) tuple from _get_local_rates. Single source of the dynamics."""
    P = 0.5
    R_SS_vals, R_LL_vals = local_rates
    p = np.empty(len(choices))
    for t in range(len(choices)):
        p[t] = logistic(beta * (P - 0.5))
        r_ss = R_SS_vals[t]; r_ll = R_LL_vals[t]
        dP = k * r_ss * (1 - P) - k * r_ll * P
        P = float(np.clip(P + dP, 0.001, 0.999))
    return p


def _kinetic_nll(choices, local_rates, k, beta):
    return bernoulli_nll_acc(choices, kinetic_pss(choices, local_rates, k, beta))


def _fit_kinetic(sdf, choices, n_starts, window):
    local_rates = _get_local_rates(sdf, window)
    def f(p):
        k = np.exp(p[0]); b = np.exp(p[1])
        return _kinetic_nll(choices, local_rates, k, b)[0]
    best = _multi_start(f, n_starts, [(-6, 2), (-2, 5)])
    k = np.exp(best.x[0]); b = np.exp(best.x[1])
    _, acc = _kinetic_nll(choices, local_rates, k, b)
    return {"nll": best.fun, "n_params": 2, "accuracy": acc,
            "params": {"k": k, "beta": b}}


# ---- Behavioral momentum ----

def momentum_pss(choices, rewards, phases, c, d, beta):
    """Per-trial P(choose SS) for behavioral momentum. `phases` is the condition
    label per trial. Single source of the dynamics."""
    r_SS_acc = 0.0; r_LL_acc = 0.0; n_SS = 0; n_LL = 0
    current_phase = phases[0]; phase_start_t = 0
    str_SS = 0.0; str_LL = 0.0
    p = np.empty(len(choices))

    for t in range(len(choices)):
        if phases[t] != current_phase:
            r_SS = r_SS_acc / max(n_SS, 1)
            r_LL = r_LL_acc / max(n_LL, 1)
            str_SS = r_SS; str_LL = r_LL
            r_SS_acc = 0.0; r_LL_acc = 0.0; n_SS = 0; n_LL = 0
            current_phase = phases[t]; phase_start_t = t

        tip = t - phase_start_t + 1
        decay_SS = 10 ** ((-tip * (c + d * str_SS)) / max(abs(str_SS) ** 0.5, 0.01))
        decay_LL = 10 ** ((-tip * (c + d * str_LL)) / max(abs(str_LL) ** 0.5, 0.01))
        curr_SS = r_SS_acc / max(n_SS, 1) if n_SS > 0 else 0.0
        curr_LL = r_LL_acc / max(n_LL, 1) if n_LL > 0 else 0.0
        pref_SS = curr_SS + str_SS * decay_SS
        pref_LL = curr_LL + str_LL * decay_LL
        p[t] = logistic(beta * (pref_SS - pref_LL))

        if choices[t] == 1:
            r_SS_acc += rewards[t]; n_SS += 1
        else:
            r_LL_acc += rewards[t]; n_LL += 1
    return p


def _momentum_nll(choices, rewards, phases, c, d, beta):
    return bernoulli_nll_acc(choices, momentum_pss(choices, rewards, phases, c, d, beta))


def _fit_momentum(choices, rewards, phases, n_starts):
    def f(p):
        c = np.exp(p[0]); d = np.exp(p[1]); b = np.exp(p[2])
        return _momentum_nll(choices, rewards, phases, c, d, b)[0]
    best = _multi_start(f, n_starts, [(-4, 2), (-6, 0), (-2, 5)])
    c = np.exp(best.x[0]); d = np.exp(best.x[1]); b = np.exp(best.x[2])
    _, acc = _momentum_nll(choices, rewards, phases, c, d, b)
    return {"nll": best.fun, "n_params": 3, "accuracy": acc,
            "params": {"c": c, "d": d, "beta": b}}


# ---- Hill climbing ----

def _zscore(x):
    """Standardize to zero mean / unit variance; returns zeros if no spread."""
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def _hill_climbing_components(choices, rewards, ici_s):
    """Replay the choice sequence and return the two decision-variable components
    per trial (pre-update state), each standardized to unit variance within the
    run. The components are:
      time-difference component:   t_LL - t_SS   (time since each option chosen)
      reward-recency component:    1/T_SS - 1/T_LL (recency of each option's reward)
    Standardizing each component removes the unit-scale mismatch (seconds vs.
    reciprocal seconds) so neither dominates the decision variable by scale alone
    and beta can calibrate the predicted probabilities off the 0/1 bounds.
    """
    n = len(choices)
    tcomp = np.empty(n); rcomp = np.empty(n)
    t_SS = 1.0; t_LL = 1.0; T_SS = 10.0; T_LL = 10.0
    for t in range(n):
        tcomp[t] = t_LL - t_SS
        rcomp[t] = 1.0 / max(T_SS, 0.01) - 1.0 / max(T_LL, 0.01)
        dt = ici_s[t] if (t < len(ici_s) and not np.isnan(ici_s[t])) else 1.0
        if choices[t] == 1:
            t_SS = dt; t_LL += dt
        else:
            t_LL = dt; t_SS += dt
        T_SS += dt; T_LL += dt
        # treat outcome > 0 as a reinforcer
        if choices[t] == 1 and rewards[t] > 0:
            T_SS = dt
        elif choices[t] == 0 and rewards[t] > 0:
            T_LL = dt
    return _zscore(tcomp), _zscore(rcomp)


def hill_climbing_pss(choices, rewards, ici_s, A, beta):
    """Per-trial P(choose SS) for the rescaled hill-climbing model. Single source
    of the dynamics used by both the fitter and predictions.py. A weights the
    reward-recency component relative to the time-difference component; beta
    scales the combined (unit-variance) decision variable."""
    tz, rz = _hill_climbing_components(choices, rewards, ici_s)
    return logistic(beta * (tz + A * rz))


def _hill_climbing_nll(choices, tz, rz, A, beta):
    # NLL kept vectorized to match the committed hill_climbing fits exactly.
    p_ss = logistic(beta * (tz + A * rz))
    p_actual = np.where(choices == 1, p_ss, 1.0 - p_ss)
    nll = -np.sum(np.log(np.clip(p_actual, 1e-10, 1.0)))
    correct = int(np.sum((p_ss >= 0.5) == (choices == 1)))
    return float(nll), correct / len(choices)


def _fit_hill_climbing(choices, rewards, ici_s, n_starts):
    if ici_s is None or not np.any(np.isfinite(ici_s)):
        return None
    tz, rz = _hill_climbing_components(choices, rewards, ici_s)
    def f(p):
        A = np.exp(p[0]); b = np.exp(p[1])
        return _hill_climbing_nll(choices, tz, rz, A, b)[0]
    best = _multi_start(f, n_starts, [(-4, 4), (-2, 5)])
    A = np.exp(best.x[0]); b = np.exp(best.x[1])
    _, acc = _hill_climbing_nll(choices, tz, rz, A, b)
    return {"nll": best.fun, "n_params": 2, "accuracy": acc,
            "params": {"A_recency": A, "beta": b}}


# ---- Ratio invariance ----

def ratio_invariance_pss(choices, rewards, omega, beta):
    """Per-trial P(choose SS) for ratio invariance. Single source of the dynamics."""
    alpha_lr = 0.1
    R_SS, R_LL = 0.0, 0.0
    eps = 1e-6
    p = np.empty(len(choices))
    for t in range(len(choices)):
        denom = R_SS + R_LL - 2 * omega
        if abs(denom) < eps:
            s_star = 0.5
        else:
            s_star = float(np.clip((R_SS - omega) / denom, 0.01, 0.99))
        p[t] = logistic(beta * (s_star - 0.5))
        if choices[t] == 1:
            R_SS = (1 - alpha_lr) * R_SS + alpha_lr * rewards[t]
        else:
            R_LL = (1 - alpha_lr) * R_LL + alpha_lr * rewards[t]
    return p


def _ratio_invariance_nll(choices, rewards, omega, beta):
    return bernoulli_nll_acc(choices, ratio_invariance_pss(choices, rewards, omega, beta))


def _fit_ratio_invariance(choices, rewards, n_starts):
    def f(p):
        omega = _sigmoid(p[0]) * 0.5
        beta = np.exp(p[1])
        return _ratio_invariance_nll(choices, rewards, omega, beta)[0]
    best = _multi_start(f, n_starts, [(-5, 5), (-2, 5)])
    omega = _sigmoid(best.x[0]) * 0.5
    beta = np.exp(best.x[1])
    _, acc = _ratio_invariance_nll(choices, rewards, omega, beta)
    return {"nll": best.fun, "n_params": 2, "accuracy": acc,
            "params": {"omega": omega, "beta": beta}}


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
