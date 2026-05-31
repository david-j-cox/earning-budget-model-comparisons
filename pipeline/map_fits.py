"""Maximum a posteriori (MAP) fits with weakly-informative priors.

A computationally lighter alternative to full hierarchical Bayes. For each
per-subject fit we add a log-prior penalty to the NLL, pulling estimates
inward and reducing boundary pileup observed in unregularized MLE.

Priors used:
  - α and related (0, 1) probabilities  : Beta(2, 2)  weakly favors mid-range
  - β and related positive rate params  : Gamma(2, 1) weakly favors moderate β

This addresses R1/R2/R4's concern that per-subject MLE on 15-80 trials
produces unidentified parameter estimates. MAP with these priors will:
  (a) pull α away from the (0, 1) boundaries unless the data strongly say
      otherwise
  (b) regularize β toward physically reasonable softmax temperatures
  (c) keep the per-subject inferential structure (one fit per subject)
      without the MCMC compute cost of full hierarchical Bayes.
"""

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import minimize

from . import config

warnings.simplefilter("ignore")


# ---- Prior log densities ----

def _log_prior_alpha(alpha):
    """Beta(2, 2) on (0, 1)."""
    if not (0 < alpha < 1):
        return -np.inf
    return stats.beta.logpdf(alpha, 2, 2)


def _log_prior_beta(beta):
    """Gamma(2, 1) on (0, ∞)."""
    if beta <= 0:
        return -np.inf
    return stats.gamma.logpdf(beta, 2, scale=1)


def _log_prior_forget(forget):
    """Beta(2, 2) on (0, 1)."""
    if not (0 < forget < 1):
        return -np.inf
    return stats.beta.logpdf(forget, 2, 2)


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _sigmoid(x):
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
        best = minimize(func, np.zeros(len(bounds)), method="L-BFGS-B", bounds=bounds)
    return best


# ---- MAP fits for the six headline models ----

def _q_learning_map(choices, rewards, valid, n_starts=10):
    """Q-learning with Beta(2,2) prior on α and Gamma(2,1) prior on β."""
    def neg_log_post(params):
        alpha = _sigmoid(params[0])
        beta = np.exp(params[1])
        # Negative log likelihood
        Q = np.array([0.0, 0.0])
        nll = 0.0
        for t in range(len(choices)):
            p_ss = _logistic(beta * (Q[1] - Q[0]))
            p_act = p_ss if choices[t] == 1 else (1 - p_ss)
            nll -= np.log(max(p_act, 1e-10))
            if valid[t]:
                Q[choices[t]] += alpha * (rewards[t] - Q[choices[t]])
        # Negative log prior
        nlp = -_log_prior_alpha(alpha) - _log_prior_beta(beta)
        return nll + nlp

    best = _multi_start(neg_log_post, n_starts, [(-5, 5), (-2, 5)])
    alpha = _sigmoid(best.x[0]); beta = np.exp(best.x[1])
    return {"alpha": alpha, "beta": beta, "neg_log_post": best.fun}


def _q_dual_alpha_map(choices, rewards, valid, n_starts=10):
    def neg_log_post(params):
        ap = _sigmoid(params[0]); an = _sigmoid(params[1])
        beta = np.exp(params[2])
        Q = np.array([0.0, 0.0])
        nll = 0.0
        for t in range(len(choices)):
            p_ss = _logistic(beta * (Q[1] - Q[0]))
            p_act = p_ss if choices[t] == 1 else (1 - p_ss)
            nll -= np.log(max(p_act, 1e-10))
            if valid[t]:
                rpe = rewards[t] - Q[choices[t]]
                a = ap if rpe > 0 else an
                Q[choices[t]] += a * rpe
        nlp = -_log_prior_alpha(ap) - _log_prior_alpha(an) - _log_prior_beta(beta)
        return nll + nlp

    best = _multi_start(neg_log_post, n_starts, [(-5, 5), (-5, 5), (-2, 5)])
    return {
        "alpha_pos": _sigmoid(best.x[0]),
        "alpha_neg": _sigmoid(best.x[1]),
        "beta": np.exp(best.x[2]),
        "neg_log_post": best.fun,
    }


def _q_forgetting_map(choices, rewards, valid, n_starts=10):
    def neg_log_post(params):
        alpha = _sigmoid(params[0]); beta = np.exp(params[1])
        forget = _sigmoid(params[2])
        Q = np.array([0.0, 0.0])
        nll = 0.0
        for t in range(len(choices)):
            p_ss = _logistic(beta * (Q[1] - Q[0]))
            p_act = p_ss if choices[t] == 1 else (1 - p_ss)
            nll -= np.log(max(p_act, 1e-10))
            if valid[t]:
                ch = choices[t]; unch = 1 - ch
                Q[ch] += alpha * (rewards[t] - Q[ch])
                Q[unch] += forget * (0.0 - Q[unch])
        nlp = -_log_prior_alpha(alpha) - _log_prior_beta(beta) - _log_prior_forget(forget)
        return nll + nlp

    best = _multi_start(neg_log_post, n_starts, [(-5, 5), (-2, 5), (-5, 5)])
    return {
        "alpha": _sigmoid(best.x[0]),
        "beta": np.exp(best.x[1]),
        "forget": _sigmoid(best.x[2]),
        "neg_log_post": best.fun,
    }


def _q_condition_aware_map(choices, rewards, valid, conds, n_starts=10):
    unique_conds = sorted(set(conds))
    n_alphas = len(unique_conds)
    cond_to_idx = {c: i for i, c in enumerate(unique_conds)}

    def neg_log_post(params):
        alphas = {c: _sigmoid(params[cond_to_idx[c]]) for c in unique_conds}
        beta = np.exp(params[n_alphas])
        Q = np.array([0.0, 0.0])
        nll = 0.0
        for t in range(len(choices)):
            p_ss = _logistic(beta * (Q[1] - Q[0]))
            p_act = p_ss if choices[t] == 1 else (1 - p_ss)
            nll -= np.log(max(p_act, 1e-10))
            if valid[t]:
                a = alphas[conds[t]]
                Q[choices[t]] += a * (rewards[t] - Q[choices[t]])
        nlp = -_log_prior_beta(beta)
        for a in alphas.values():
            nlp -= _log_prior_alpha(a)
        return nll + nlp

    bounds = [(-5, 5)] * n_alphas + [(-2, 5)]
    best = _multi_start(neg_log_post, n_starts, bounds)
    out = {f"alpha_{c}": _sigmoid(best.x[cond_to_idx[c]]) for c in unique_conds}
    out["beta"] = np.exp(best.x[n_alphas])
    out["neg_log_post"] = best.fun
    return out


def _q_table_map(choices, rewards, valid, states, n_starts=10):
    def neg_log_post(params):
        alpha = _sigmoid(params[0]); beta = np.exp(params[1])
        Q = {}
        nll = 0.0
        for t, s in enumerate(states):
            q = Q.setdefault(s, np.array([0.0, 0.0]))
            p_ss = _logistic(beta * (q[1] - q[0]))
            p_act = p_ss if choices[t] == 1 else (1 - p_ss)
            nll -= np.log(max(p_act, 1e-10))
            if valid[t]:
                q[choices[t]] += alpha * (rewards[t] - q[choices[t]])
        nlp = -_log_prior_alpha(alpha) - _log_prior_beta(beta)
        return nll + nlp

    best = _multi_start(neg_log_post, n_starts, [(-5, 5), (-2, 5)])
    return {
        "alpha": _sigmoid(best.x[0]),
        "beta": np.exp(best.x[1]),
        "neg_log_post": best.fun,
    }


# ---- Per-subject orchestration ----

MODELS = ["q_learning", "q_dual_alpha", "q_forgetting", "q_condition_aware",
          "q_table_operant", "q_table_operant_budget"]


def fit_one_subject_map(subject, sdf, n_starts=10):
    """Run all six MAP fits on one subject's pooled-across-conditions data."""
    from .feature_engineering import state_operant, state_operant_budget

    sdf = sdf.sort_values(["condition", "trial"]).reset_index(drop=True)
    choices = sdf["choice"].values.astype(int)
    outcomes = sdf["outcome_norm"].values.astype(float)
    valid = ~np.isnan(outcomes)
    outcomes_filled = np.where(valid, outcomes, 0.0)
    conds = sdf["condition"].values
    n_obs = len(choices)

    rows = []
    base = {
        "subject": subject,
        "group": sdf["group"].iloc[0],
        "n_obs": n_obs,
    }

    # Q-learning
    r = _q_learning_map(choices, outcomes_filled, valid, n_starts)
    rows.append({**base, "model": "q_learning",
                 "param_alpha": r["alpha"], "param_beta": r["beta"],
                 "neg_log_post": r["neg_log_post"]})
    # Dual α
    r = _q_dual_alpha_map(choices, outcomes_filled, valid, n_starts)
    rows.append({**base, "model": "q_dual_alpha",
                 "param_alpha_pos": r["alpha_pos"], "param_alpha_neg": r["alpha_neg"],
                 "param_beta": r["beta"], "neg_log_post": r["neg_log_post"]})
    # Forgetting
    r = _q_forgetting_map(choices, outcomes_filled, valid, n_starts)
    rows.append({**base, "model": "q_forgetting",
                 "param_alpha": r["alpha"], "param_beta": r["beta"],
                 "param_forget": r["forget"], "neg_log_post": r["neg_log_post"]})
    # Condition aware
    r = _q_condition_aware_map(choices, outcomes_filled, valid, conds, n_starts)
    row = {**base, "model": "q_condition_aware",
           "param_beta": r["beta"], "neg_log_post": r["neg_log_post"]}
    for k, v in r.items():
        if k.startswith("alpha_"):
            row[f"param_{k}"] = v
    rows.append(row)
    # Q-table operant
    states_op = [state_operant(row) for _, row in sdf.iterrows()]
    r = _q_table_map(choices, outcomes_filled, valid, states_op, n_starts)
    rows.append({**base, "model": "q_table_operant",
                 "param_alpha": r["alpha"], "param_beta": r["beta"],
                 "neg_log_post": r["neg_log_post"]})
    # Q-table operant + budget
    states_opb = [state_operant_budget(row) for _, row in sdf.iterrows()]
    r = _q_table_map(choices, outcomes_filled, valid, states_opb, n_starts)
    rows.append({**base, "model": "q_table_operant_budget",
                 "param_alpha": r["alpha"], "param_beta": r["beta"],
                 "neg_log_post": r["neg_log_post"]})

    return rows


def run_all_map(features_df, n_starts=10, n_jobs=None, verbose=True):
    n_jobs = n_jobs or config.N_JOBS
    subjects = features_df["subject"].unique()
    if verbose:
        print(f"MAP fitting {len(subjects)} subjects × {len(MODELS)} models "
              f"({n_jobs} workers, n_starts={n_starts})")

    def _do(sub):
        sdf = features_df[features_df["subject"] == sub]
        return fit_one_subject_map(sub, sdf, n_starts=n_starts)

    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(_do)(sub) for sub in subjects
    )
    rows = [r for sub_rows in results for r in sub_rows]
    return pd.DataFrame(rows)


def main():
    import time
    features = pd.read_parquet(config.TRANSFORMED_DIR / "exp1_features.parquet")
    t0 = time.time()
    map_df = run_all_map(features, verbose=True)
    out = config.RESULTS_DIR / "fits_map.parquet"
    map_df.to_parquet(out, index=False)
    print(f"\nSaved {len(map_df)} MAP fit rows to {out.name}")
    print(f"Elapsed: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
