"""Pietras et al. (2003) energy-budget model — four operationalizations.

The energy-budget framework (Pietras & Hackenberg, 2001; Pietras et al., 2003)
formalizes choice between variable and certain options as a function of whether
the organism is on pace to meet the resource requirement R within the available
time T at average rate t (T > Rt = positive budget; T < Rt = negative budget).

Under a positive budget, lower-variance options are preferred (a certain rate
suffices). Under a negative budget, higher-variance options are preferred
because variance is the only way to exceed the deterministic shortfall.

For this task:
  - SS is the lower-variance option (1-second deterministic payoff).
  - LL is the higher-variance option (adjusting-delay payoff; variability in
    timing relative to the constant rate of loss makes it the variable option).
  - Programmed loss rates come from config.CONDITIONS (Table 1).

Six operationalizations are implemented (four energy-budget, two marginal-value):

1. energy_budget          : Minimal one-parameter sigmoid, P(SS) = sigmoid(beta * d).
                            (The canonical "as published" baseline.)
2. energy_budget_threshold: Two-parameter sigmoid with learnable threshold tau,
                            P(SS) = sigmoid(beta * (d - tau)).
3. energy_budget_zscore   : Caraco-style z-score risk-sensitivity. For each option,
                            compute z_i = (mu_i - R_hat) / sigma_i; choose option
                            with higher z. Uses rolling option-specific reward
                            mean and SD. One free parameter (softmax beta).
4. energy_budget_categorical: Non-parametric categorical preference-reversal test.
                            Reports per-subject P(LL | negative budget) -
                            P(LL | positive budget). Not a likelihood fit.

The task is a binary smaller-sooner / larger-later choice rather than a patch-
residence task, so the marginal value theorem (Charnov, 1976) is operationalized
as the short-term rate-maximization / opportunity-cost rule it implies (Stephens
& Anderson, 2001; Stephens, 2002): each option's value is its magnitude minus the
opportunity cost of its delay valued at the background environmental rate rho,
SV_i = V_i - rho * D_i, and the organism prefers the option with the higher
rate-adjusted value. The decision variable is

    d(rho) = SV_SS - SV_LL = (V_SS - V_LL) + rho * (D_LL - D_SS),
    P(SS)  = sigmoid(beta * d / scale),

with V_SS = 1 (the normalized smaller-sooner amount), V_LL = the group LL:SS ratio
(5 or 10), D_SS = 1 s (the deterministic SS delay), and D_LL the adjusting delay
to LL. Two variants differ only in how the background rate rho is set:

5. mvt           : rho fixed to the condition loss rate (the rate at which points
                   drain), so the opportunity cost of waiting D_LL seconds is the
                   points forgone to the budget drain. One free parameter (beta).
6. mvt_fitted_rho: rho is a free per-subject opportunity cost (the revealed
                   background rate). Two free parameters (beta, rho).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .. import config


# ===========================================================================
# Public API
# ===========================================================================

def fit_energy_budget(sdf: pd.DataFrame, n_starts: int = 10) -> list[dict]:
    """Fit all six energy-budget-family operationalizations on one subject."""
    n_obs = len(sdf)
    if n_obs < 2:
        return []

    rows = []
    rows.append(_fit_minimal_sigmoid(sdf, n_starts))
    rows.append(_fit_threshold_sigmoid(sdf, n_starts))
    rows.append(_fit_zscore(sdf, n_starts))
    cat = _categorical_test(sdf)
    if cat is not None:
        rows.append(cat)
    rows.append(_fit_mvt(sdf, n_starts))
    rows.append(_fit_mvt_fitted_rho(sdf, n_starts))
    return [r for r in rows if r is not None]


# ===========================================================================
# Variant 1: Minimal one-parameter sigmoid (the published canonical)
# ===========================================================================

def _fit_minimal_sigmoid(sdf, n_starts):
    n_obs = len(sdf)
    choices = sdf["choice"].values.astype(int)
    dist = sdf["dist_to_death"].values.astype(float)
    # Rescale dist to roughly unit scale so beta lives in a sensible range
    scale = max(np.nanmax(np.abs(dist)), 1.0)
    d = dist / scale

    def neg_log_lik(params):
        beta = params[0]
        return _sigmoid_nll(choices, d, beta)

    # β scale is now on rescaled distance; wider bounds.
    best = _multi_start(neg_log_lik, n_starts, [(-10.0, 10.0)])
    beta = float(best.x[0])
    nll = float(best.fun)
    acc = _sigmoid_accuracy(choices, d, beta)
    return _row("energy_budget", 1, nll, n_obs, acc, beta=beta, scale=scale)


# ===========================================================================
# Variant 2: Two-parameter sigmoid with learnable threshold
# ===========================================================================

def _fit_threshold_sigmoid(sdf, n_starts):
    n_obs = len(sdf)
    choices = sdf["choice"].values.astype(int)
    dist = sdf["dist_to_death"].values.astype(float)
    scale = max(np.nanmax(np.abs(dist)), 1.0)
    d = dist / scale

    def neg_log_lik(params):
        beta, tau = params
        return _sigmoid_threshold_nll(choices, d, beta, tau)

    best = _multi_start(neg_log_lik, n_starts, [(-10.0, 10.0), (-2.0, 2.0)])
    beta = float(best.x[0])
    tau = float(best.x[1])
    nll = float(best.fun)
    acc = _sigmoid_threshold_accuracy(choices, d, beta, tau)
    return _row("energy_budget_threshold", 2, nll, n_obs, acc,
                beta=beta, tau=tau, scale=scale)


# ===========================================================================
# Variant 3: Caraco-style z-score risk-sensitivity
# ===========================================================================

def _fit_zscore(sdf, n_starts):
    """Per-trial z-score for each option:
        z_i = (mu_i - R_hat) / sigma_i
    where mu_i is rolling per-option mean reward, sigma_i is rolling per-option SD,
    and R_hat is the per-trial required rate from the budget arithmetic.

    P(SS) = sigmoid(beta * (z_SS - z_LL)).
    """
    n_obs = len(sdf)
    if n_obs < 5:  # need a few trials per option for rolling stats
        return None
    choices = sdf["choice"].values.astype(int)
    z_diff = _compute_zscore_diff(sdf)

    def neg_log_lik(params):
        beta = params[0]
        return _sigmoid_nll(choices, z_diff, beta)

    best = _multi_start(neg_log_lik, n_starts, [(-10.0, 10.0)])
    beta = float(best.x[0])
    nll = float(best.fun)
    acc = _sigmoid_accuracy(choices, z_diff, beta)
    return _row("energy_budget_zscore", 1, nll, n_obs, acc, beta=beta)


def _compute_zscore_diff(sdf):
    """Per-trial z_SS - z_LL using rolling per-option statistics."""
    n = len(sdf)
    choices = sdf["choice"].values.astype(int)
    outcomes = sdf["outcome"].fillna(0).values.astype(float)
    bank = sdf["bank"].values.astype(float)
    time_left = sdf["time_left"].values.astype(float)
    cond = sdf["condition"].values

    z_diff = np.zeros(n)
    # Maintain running stats per option using Welford's running mean/variance
    n_ss = 0; mu_ss = 0.0; m2_ss = 0.0
    n_ll = 0; mu_ll = 0.0; m2_ll = 0.0
    eps = 1e-6

    for t in range(n):
        # Per-trial required earning rate: how many points per second still
        # needed to satisfy the budget? Derived from programmed values.
        R = config.CONDITIONS[cond[t]]["R"]
        loss_rate = config.CONDITIONS[cond[t]]["loss_rate"]
        tl = max(time_left[t], eps)
        # Required net rate = (R - bank) / time_left + loss_rate
        # (need to earn enough to cover deficit AND ongoing loss)
        deficit = max(R - bank[t], 0.0)
        r_hat = deficit / tl + loss_rate
        # Default mu/sigma if option not yet observed
        mu_ss_est = mu_ss if n_ss > 0 else 0.0
        sd_ss_est = np.sqrt(m2_ss / max(n_ss - 1, 1)) if n_ss > 1 else 1.0
        mu_ll_est = mu_ll if n_ll > 0 else 0.0
        sd_ll_est = np.sqrt(m2_ll / max(n_ll - 1, 1)) if n_ll > 1 else 1.0
        z_ss = (mu_ss_est - r_hat) / max(sd_ss_est, eps)
        z_ll = (mu_ll_est - r_hat) / max(sd_ll_est, eps)
        z_diff[t] = z_ss - z_ll
        # Update running stats with observed outcome (Welford)
        if choices[t] == 1:
            n_ss += 1
            delta = outcomes[t] - mu_ss
            mu_ss += delta / n_ss
            m2_ss += delta * (outcomes[t] - mu_ss)
        else:
            n_ll += 1
            delta = outcomes[t] - mu_ll
            mu_ll += delta / n_ll
            m2_ll += delta * (outcomes[t] - mu_ll)
    # Rescale to unit range for numerical stability of softmax
    scale = max(np.nanmax(np.abs(z_diff)), 1.0)
    return z_diff / scale


# ===========================================================================
# Variant 4: Categorical preference-reversal test (non-parametric)
# ===========================================================================

def _categorical_test(sdf):
    """Per-subject directional test: P(LL | negative budget) vs. P(LL | positive).

    Returns a row with a 'param_delta' field giving the within-subject difference.
    Not a likelihood fit — no AIC/BIC. NLL is set so the model contributes a
    chance-level prediction at the trial level (for parity in the comparison table).
    """
    pos_mask = sdf["budget_sign"].values == "positive"
    neg_mask = sdf["budget_sign"].values == "negative"
    n_obs = len(sdf)
    if pos_mask.sum() < 2 or neg_mask.sum() < 2:
        return None
    p_ll_pos = float(1 - sdf.loc[pos_mask, "choice"].mean())
    p_ll_neg = float(1 - sdf.loc[neg_mask, "choice"].mean())
    delta = p_ll_neg - p_ll_pos  # positive = framework prediction holds
    # Use the per-condition base rates to make a per-trial prediction.
    p_ll_per_trial = np.where(pos_mask, p_ll_pos, p_ll_neg)
    p_ss_per_trial = 1 - p_ll_per_trial
    choices = sdf["choice"].values.astype(int)
    p_actual = np.where(choices == 1, p_ss_per_trial, p_ll_per_trial)
    nll = float(-np.sum(np.log(np.clip(p_actual, 1e-10, 1.0))))
    acc = float(((p_ss_per_trial >= 0.5).astype(int) == choices).mean())
    return _row("energy_budget_categorical", 2, nll, n_obs, acc,
                delta=delta, p_ll_pos=p_ll_pos, p_ll_neg=p_ll_neg)


# ===========================================================================
# Variants 5-6: Marginal value theorem (short-term rate maximization)
# ===========================================================================

def _mvt_components(sdf):
    """Magnitude/delay components of the MVT decision variable.

    Returns (A, B) arrays such that the rate-adjusted SS-vs-LL decision variable
    is d(rho) = A + rho * B, where
        A = V_SS - V_LL      (the magnitude advantage of SS; <= 0 here)
        B = D_LL - D_SS      (the extra delay incurred by waiting for LL; >= 0).
    V_SS = 1, V_LL = the group LL:SS ratio (5 or 10), D_SS = 1 s, D_LL = delay_ll.
    """
    ratio = sdf["group"].map(config.RATIO_BY_GROUP).values.astype(float)
    A = 1.0 - ratio
    B = sdf["delay_ll"].values.astype(float) - 1.0
    return A, B


def _fit_mvt(sdf, n_starts):
    """MVT with the background rate rho fixed to the condition loss rate."""
    n_obs = len(sdf)
    choices = sdf["choice"].values.astype(int)
    A, B = _mvt_components(sdf)
    rho = sdf["loss_rate"].values.astype(float)
    d_raw = A + rho * B
    scale = max(np.nanmax(np.abs(d_raw)), 1.0)
    d = d_raw / scale

    def neg_log_lik(params):
        beta = params[0]
        return _sigmoid_nll(choices, d, beta)

    best = _multi_start(neg_log_lik, n_starts, [(-10.0, 10.0)])
    beta = float(best.x[0])
    nll = float(best.fun)
    acc = _sigmoid_accuracy(choices, d, beta)
    return _row("mvt", 1, nll, n_obs, acc, beta=beta, scale=scale)


def _fit_mvt_fitted_rho(sdf, n_starts):
    """MVT with the background rate rho estimated as a free per-subject parameter.

    The decision variable d(rho) = A + rho * B is divided by a fixed,
    rho-independent normalizer (computed at the condition loss rate) so that beta
    stays on a comparable scale to the other family members and prediction replay
    is exact given the stored scale.
    """
    n_obs = len(sdf)
    choices = sdf["choice"].values.astype(int)
    A, B = _mvt_components(sdf)
    rho_ref = sdf["loss_rate"].values.astype(float)
    scale = max(np.nanmax(np.abs(A + rho_ref * B)), 1.0)

    def neg_log_lik(params):
        beta, rho = params
        d = (A + rho * B) / scale
        return _sigmoid_nll(choices, d, beta)

    best = _multi_start(neg_log_lik, n_starts, [(-10.0, 10.0), (0.0, 5.0)])
    beta = float(best.x[0])
    rho = float(best.x[1])
    nll = float(best.fun)
    d = (A + rho * B) / scale
    acc = _sigmoid_accuracy(choices, d, beta)
    return _row("mvt_fitted_rho", 2, nll, n_obs, acc, beta=beta, rho=rho, scale=scale)


# ===========================================================================
# Shared helpers
# ===========================================================================

def _sigmoid_nll(choices, x, beta):
    p_ss = _logistic(beta * x)
    p_actual = np.where(choices == 1, p_ss, 1 - p_ss)
    return -np.sum(np.log(np.clip(p_actual, 1e-10, 1.0)))


def _sigmoid_accuracy(choices, x, beta):
    p_ss = _logistic(beta * x)
    preds = (p_ss >= 0.5).astype(int)
    return float((preds == choices).mean())


def _sigmoid_threshold_nll(choices, x, beta, tau):
    p_ss = _logistic(beta * (x - tau))
    p_actual = np.where(choices == 1, p_ss, 1 - p_ss)
    return -np.sum(np.log(np.clip(p_actual, 1e-10, 1.0)))


def _sigmoid_threshold_accuracy(choices, x, beta, tau):
    p_ss = _logistic(beta * (x - tau))
    preds = (p_ss >= 0.5).astype(int)
    return float((preds == choices).mean())


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _row(model, n_params, nll, n_obs, acc, **params):
    aic = 2 * n_params + 2 * nll
    bic = n_params * np.log(max(n_obs, 1)) + 2 * nll
    denom = n_obs - n_params - 1
    aicc = aic + 2 * n_params * (n_params + 1) / denom if denom > 0 else np.nan
    row = {
        "model_family": "energy_budget",
        "model": model,
        "n_params": n_params,
        "n_obs": n_obs,
        "nll": nll,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "accuracy": acc,
    }
    for k, v in params.items():
        row[f"param_{k}"] = v
    return row


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
        best = minimize(func, np.zeros(len(bounds)), method="L-BFGS-B", bounds=bounds)
    return best
