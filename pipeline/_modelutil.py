"""Shared numerics for model fitting and prediction.

Single source of truth for the logistic choice link and the Bernoulli
negative-log-likelihood / accuracy reducer. Each model defines its per-trial
P(choose SS) sequence in exactly one place (its model module); the fitter wraps
that sequence with `bernoulli_nll_acc`, and `predictions.py` replays the same
sequence. This prevents the fit and prediction paths from drifting apart.

`bernoulli_nll_acc` is intentionally loop-based: it preserves the exact
floating-point summation order of the original per-trial accumulation, so
refactoring the dynamics into shared functions does not perturb the multi-start
MLE results.
"""

import numpy as np


def logistic(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def bernoulli_nll_acc(choices, p_ss):
    """Negative log-likelihood and argmax accuracy of a 0/1 choice vector under
    per-trial P(choice == 1) = p_ss.

    Returns (nll, accuracy). Loop-based to preserve summation order; matches the
    original inline accumulation `nll -= log(max(p_actual, 1e-10))` exactly.
    """
    nll = 0.0
    correct = 0
    for t in range(len(choices)):
        pa = p_ss[t] if choices[t] == 1 else (1.0 - p_ss[t])
        nll -= np.log(max(pa, 1e-10))
        if (p_ss[t] >= 0.5) == (choices[t] == 1):
            correct += 1
    return nll, correct / len(choices)
