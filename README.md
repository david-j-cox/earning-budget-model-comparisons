# Cross-tradition model comparison for earning-budget choice (ML companion)

Trial-by-trial model comparison of human choice in an experiential delay/
probability discounting earning-budget task (Experiment 1). Twenty-four models
spanning six theoretical traditions are fit per participant and compared on
predictive fit (AICc), per-trial calibration, and per-subject win rates, with
parameter- and model-recovery checks.

## Task and data

Participants chose repeatedly between a smaller-sooner (SS) and larger-later
(LL) option while points were continuously lost at a condition-specific rate
(an "earning budget"). Two groups differ in the LL:SS point ratio (5:1 and
10:1). Four conditions span positive to negative net budgets:

| condition | net rate (SS-only) | budget sign |
|-----------|--------------------|-------------|
| Chicken   | +0.67 pt/s         | positive    |
| Crab      | +0.50 pt/s         | positive    |
| Turtle    | -0.05 pt/s         | negative    |
| Piranha   | -0.50 pt/s         | negative    |

The programmed parameters are defined in `pipeline/config.py` (Table 2 of the
manuscript).

### Data availability

The source Exp1 spreadsheets are under `data/raw/`; stage 1 reads them and writes
the long-format and engineered-feature tables to `data/transformed/`. All fitted
results needed to reproduce the figures and statistics are committed under
`data/results/`. Note that `data/raw/` contains human-subjects data — review your
data-sharing obligations before distributing the repository publicly.

## Model families

| family (`model_family`) | models |
|-------------------------|--------|
| baseline                | random, bias, win-stay/lose-shift, logistic |
| matching                | generalized matching law |
| energy_budget           | one-parameter sigmoid, two-parameter + threshold, Caraco z-score, categorical preference-reversal, marginal value (loss-rate rho), marginal value (fitted rho) |
| historical_dynamics     | melioration, kinetic, behavioral momentum, hill-climbing, ratio invariance |
| rl_action_value         | Q-learning, dual-α, forgetting, dynamic-α, condition-aware |
| rl_state_action         | Q-table (operant state), Q-table (operant + budget state) |
| human_as_qtable         | human-as-Q-table (operant), (operant + budget) |

Each model is fit at two scopes: `all` (parameters pooled across the four
conditions) and `condition` (separate parameters per condition).

## Repository layout

```
data/
  raw/                   source Exp1 spreadsheets (Exp1_05_to_1.xlsx, Exp1_10_to_1.xlsx)
  transformed/           exp1_long.parquet, exp1_features.parquet (+ normalization variants)
  results/               fitted parquet outputs and CSVs
    figures/             all PNG figures
pipeline/                importable library package
  config.py              paths and programmed task parameters
  _modelutil.py          shared logistic link + Bernoulli NLL/accuracy reducer
  load_data.py           read Exp1 spreadsheets -> long format
  feature_engineering.py derive per-trial features (budget distance, rolling rates, states)
  models/                per-family model dynamics (single source for fit + prediction)
    baselines.py  matching.py  energy_budget.py
    historical_dynamics.py  rl_action_value.py  rl_state_action.py
  fit_runner.py          orchestrates per-subject fits across all families/scopes
  predictions.py         replay fitted models to per-trial predicted P(SS)
  map_fits.py            MAP re-fits of the six headline RL models
  winner_tallies.py      per-subject AICc winners (pooled and per-condition)
  pairwise_stats.py      pairwise paired-Wilcoxon model comparisons on AICc
  parameter_recovery.py  parameter and model recovery (16 parametric models)
  make_figures.py        calibration, trajectory, residual, and combined figures
  residual_summary.py    per-family median trial-level residual table
scripts/                 runnable entry points
  run.py                       stage 1 only (load -> features -> fits)
  run_pipeline.py              end-to-end orchestrator (all stages)
  run_normalization_sensitivity.py  refit under five normalization modes
  recompute_hill_climbing.py   one-time migration (see docstring)
  recalibration_benchmark.py   calibration diagnostics
README.md  requirements.txt  .gitignore
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running

From the repo root:

```bash
python scripts/run_pipeline.py                 # full pipeline, stage 1..8
python scripts/run_pipeline.py --from figures   # resume from a later stage
python scripts/run_pipeline.py --only residuals # run a single stage
```

Individual stages can also be run directly, e.g. `python -m pipeline.make_figures`
or `python -m pipeline.residual_summary` (from the repo root).

## Key outputs (`data/results/`)

- `fits_by_subject.parquet` — per (subject, scope, condition, model) fit row
  (nll, AIC, AICc, BIC, accuracy, parameters).
- `predictions.parquet` — per (subject, condition, trial, model) observed choice
  and predicted P(SS).
- `winners_pooled.parquet`, `winners_by_condition.parquet` — per-subject AICc winners.
- `pairwise_wilcoxon.parquet` — pairwise model comparisons (median ΔAICc, Bonferroni p).
- `residual_summary.csv` — per-family median trial-level |residual|.
- `parameter_recovery*.parquet`, `model_recovery_confusion*.csv` — recovery analyses.
- `figures/` — calibration, per-family trajectory/residual, combined 4×4 per-family,
  winner, and pairwise-heatmap figures.

## Reproducibility

Fitting uses multi-start maximum-likelihood (scipy `L-BFGS-B`, fixed seed). For
the multi-parameter, non-convex models, per-subject point estimates can differ
at the third decimal across numpy/scipy versions; this does not change any
reported aggregate (per-model median AICc is stable to ±0.003, and per-subject
AICc-winner tallies are unchanged). The committed `results/` were produced with
the dependency set in `requirements.txt`. Prediction replay is deterministic
given fitted parameters, except `logistic`, whose prediction path re-fits a
scikit-learn `LogisticRegression` and is therefore solver/version dependent at
the ~1e-4 level.
