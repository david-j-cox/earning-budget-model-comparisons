"""State-action Q-table models.

Three variants:
  - q_table_operant       : Q(s, a) over operant state (delay, prev_choice, prev_outcome_sign)
  - q_table_operant_budget: same + budget state (bank, time_left)
  - human_as_qtable       : Cox & Santos (2025) "human choice modeled by Q-learning":
                            update Q-table directly from observed (state, choice) pairs with
                            +1/-1 reward for correct/incorrect next-response prediction.
                            Evaluated in two flavors (operant only, operant+budget).

All three operate on the same discretized state representations from
feature_engineering.state_operant and state_operant_budget.

For the first two, we fit alpha and beta via MLE (softmax over Q values).
For human-as-qtable, no MLE; we measure prediction accuracy directly.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..feature_engineering import state_operant, state_operant_budget
from .._modelutil import logistic, bernoulli_nll_acc


def fit_state_action_rl(sdf: pd.DataFrame, n_starts: int = 10) -> list[dict]:
    """Fit all three state-action variants on one subject."""
    if len(sdf) < 2:
        return []

    rows = []
    for state_fn, label in [(state_operant, "q_table_operant"),
                            (state_operant_budget, "q_table_operant_budget")]:
        result = _fit_q_state_action(sdf, state_fn, n_starts)
        if result is not None:
            rows.append(_row(label, result, len(sdf)))

    # Cox & Santos protocol
    for state_fn, label in [(state_operant, "human_as_qtable_operant"),
                            (state_operant_budget, "human_as_qtable_operant_budget")]:
        result = _human_as_qtable(sdf, state_fn)
        if result is not None:
            rows.append(_row_human_qtable(label, result, len(sdf)))

    return [r for r in rows if r is not None]


def _row(model, fit, n_obs):
    nll = fit["nll"]; n_params = fit["n_params"]
    aic = 2 * n_params + 2 * nll
    bic = n_params * np.log(max(n_obs, 1)) + 2 * nll
    denom = n_obs - n_params - 1
    aicc = aic + 2 * n_params * (n_params + 1) / denom if denom > 0 else np.nan
    row = {
        "model_family": "rl_state_action",
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


def _row_human_qtable(model, result, n_obs):
    """Cox & Santos has no MLE parameters — just accuracy and # distinct states."""
    return {
        "model_family": "human_as_qtable",
        "model": model,
        "n_params": 0,                   # no fitted free parameters
        "n_obs": n_obs,
        "nll": result["nll"],
        "aic": 2 * result["nll"],
        "aicc": 2 * result["nll"],      # k=0 so correction term is 0
        "bic": 2 * result["nll"],
        "accuracy": result["accuracy"],
        "param_n_states": result["n_states"],
    }


# ---- State-action Q-learning with MLE ----

def _fit_q_state_action(sdf, state_fn, n_starts):
    """Fit alpha and beta for Q(s,a) over the given state representation."""
    # Precompute states and outcomes
    sdf = sdf.reset_index(drop=True)
    states = [state_fn(row) for _, row in sdf.iterrows()]
    choices = sdf["choice"].values.astype(int)
    outcomes = sdf["outcome_norm"].values.astype(float)
    valid_out = ~np.isnan(outcomes)
    outcomes_filled = np.where(valid_out, outcomes, 0.0)

    def f(p):
        alpha = _sigmoid(p[0]); beta = np.exp(p[1])
        nll, _ = _q_sa_loop(states, choices, outcomes_filled, valid_out, alpha, beta)
        return nll

    best = _multi_start(f, n_starts, [(-5, 5), (-2, 5)])
    alpha = _sigmoid(best.x[0]); beta = np.exp(best.x[1])
    nll, acc = _q_sa_loop(states, choices, outcomes_filled, valid_out, alpha, beta)
    return {"nll": nll, "n_params": 2, "accuracy": acc,
            "params": {"alpha": alpha, "beta": beta}}


def q_state_action_pss(states, choices, rewards, valid_out, alpha, beta):
    """Per-trial P(choose SS) for state-action Q-learning. Single source of the
    dynamics, used by both the fitter (via _q_sa_loop) and predictions.py. The
    Q-table is populated lazily as dict[state] -> [Q_LL, Q_SS]."""
    Q = {}
    p = np.empty(len(states))
    for t, s in enumerate(states):
        q = Q.setdefault(s, np.array([0.0, 0.0]))
        p[t] = logistic(beta * (q[1] - q[0]))
        if valid_out[t]:
            chosen = choices[t]
            q[chosen] += alpha * (rewards[t] - q[chosen])
    return p


def _q_sa_loop(states, choices, outcomes, valid_out, alpha, beta):
    return bernoulli_nll_acc(
        choices, q_state_action_pss(states, choices, outcomes, valid_out, alpha, beta))


# ---- Cox & Santos "human as Q-table" ----

def _human_as_qtable(sdf, state_fn):
    """Direct Q-table population from observed (state, choice) pairs.

    Per Cox & Santos (2025): on each trial, update Q[state][predicted_choice] by
    +1 if the AO predicted correctly, -1 if wrong. Here we measure how well a
    Q-table that has been populated from the participant's own history predicts
    the next response.

    Concretely (mirroring Cox & Santos' "human as Q-table" condition):
    For each trial t:
      - Use the Q-table built from trials 1..t-1 to predict choice at trial t.
      - The predicted choice is argmax_a Q[s_t][a].
      - Then update Q[s_t][observed_choice_t] += 1 (treat the observed action as
        "what was reinforced in this state").
    Returns prediction accuracy and a notional NLL assuming deterministic predictions
    with a small epsilon for ties/unseen states.
    """
    sdf = sdf.reset_index(drop=True)
    states = [state_fn(row) for _, row in sdf.iterrows()]
    choices = sdf["choice"].values.astype(int)

    Q = {}  # state -> np.array([count_LL, count_SS])
    correct = 0
    nll = 0.0
    eps = 0.05  # smoothing for prediction probability
    for t, s in enumerate(states):
        q = Q.get(s, np.array([0.0, 0.0]))
        if q[0] == q[1]:
            # Tie or unseen → predict majority class so far
            pred = int(np.mean(choices[:t]) >= 0.5) if t > 0 else 0
            p_predicted = 0.5
        else:
            pred = 1 if q[1] > q[0] else 0
            # Confidence proportional to count gap
            total = q.sum() + 1
            p_predicted = max(q[pred], 1) / total
            p_predicted = max(p_predicted, eps)
            p_predicted = min(p_predicted, 1 - eps)

        if pred == choices[t]:
            correct += 1
            nll -= np.log(p_predicted)
        else:
            nll -= np.log(1 - p_predicted)

        # Now reinforce the observed choice in this state
        q = Q.setdefault(s, np.array([0.0, 0.0]))
        q[choices[t]] += 1.0

    return {
        "nll": nll,
        "accuracy": correct / len(states),
        "n_states": len(Q),
    }


# ---- Helpers ----

def _sigmoid(x):
    """Parameter transform (unconstrained -> (0,1)) used in the fit objective."""
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
