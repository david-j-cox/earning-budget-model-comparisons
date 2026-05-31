"""End-to-end pipeline orchestrator.

Runs every stage in dependency order, from the Exp1 raw data through fits,
predictions, model-comparison statistics, recovery analyses, and figures.

Usage (from the repo root, with the environment in requirements.txt installed):
    python scripts/run_pipeline.py            # full pipeline
    python scripts/run_pipeline.py --from predictions   # resume from a later stage

Stages and their outputs (under data/results/ unless noted):
    1. fits          load_data -> feature_engineering -> fit_runner
                     -> data/transformed/exp1_features.parquet, fits_by_subject.parquet
    2. predictions   per-trial predicted P(SS) -> predictions.parquet
    3. map_fits      MAP re-fits of the six headline RL models -> fits_map.parquet
    4. winners       per-subject AICc winners -> winners_*.parquet, fig_winners_*
    5. pairwise      pairwise Wilcoxon on AICc -> pairwise_wilcoxon.parquet, heatmaps
    6. recovery      parameter/model recovery -> parameter_recovery*, model_recovery_*
    7. figures       calibration, trajectory, residual, combined -> data/results/figures/
    8. residuals     per-family residual summary -> residual_summary.csv

Reproducibility note: stage 1 uses multi-start MLE (scipy L-BFGS-B). The fitted
values for the multi-parameter, non-convex models can differ at the third
decimal across numpy/scipy versions; all reported aggregates (per-model median
AICc, winner tallies) are stable to reported precision. See README.
"""

import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # repo root, for `import pipeline`
sys.path.insert(0, str(_HERE))         # scripts/, for sibling `import run`

import argparse

import run as _fit_stage
from pipeline import (predictions, map_fits, winner_tallies, pairwise_stats,
                      parameter_recovery, make_figures, residual_summary)


STAGES = ["fits", "predictions", "map_fits", "winners", "pairwise",
          "recovery", "figures", "residuals"]

STAGE_FNS = {
    "fits": _fit_stage.main,
    "predictions": predictions.main,
    "map_fits": map_fits.main,
    "winners": winner_tallies.main,
    "pairwise": pairwise_stats.main,
    "recovery": parameter_recovery.main,
    "figures": make_figures.main,
    "residuals": residual_summary.main,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="start", choices=STAGES, default="fits",
                    help="resume from this stage (default: fits)")
    ap.add_argument("--only", choices=STAGES,
                    help="run only this single stage")
    args = ap.parse_args()

    if args.only:
        stages = [args.only]
    else:
        stages = STAGES[STAGES.index(args.start):]

    for stage in stages:
        print("\n" + "=" * 70)
        print(f"STAGE: {stage}")
        print("=" * 70)
        STAGE_FNS[stage]()
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
