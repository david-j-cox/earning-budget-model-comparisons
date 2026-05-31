"""Outcome normalization sensitivity analysis.

Refits the full 23-model pipeline under each of five outcome-normalization modes
and saves results to mode-specific parquet files. Lets us check whether the AICc
ranking depends on the normalization choice (R1/R2/R4 critique).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import time
import pandas as pd

from pipeline import config, load_data, feature_engineering, fit_runner
from pipeline.feature_engineering import NORMALIZATION_MODES


def main():
    t0 = time.time()

    # Load raw once
    print("Loading raw Exp1 data ...")
    long_df = load_data.load_exp1(verbose=False)
    print(f"  {len(long_df)} trials across {long_df['subject'].nunique()} subjects.\n")

    for mode in NORMALIZATION_MODES:
        print("=" * 60)
        print(f"Normalization mode: {mode}")
        print("=" * 60)
        t_mode = time.time()
        features_df = feature_engineering.build_features(long_df, normalization_mode=mode)
        # Save per-mode features for downstream prediction generation
        features_path = config.TRANSFORMED_DIR / f"exp1_features__{mode}.parquet"
        features_df.to_parquet(features_path, index=False)
        print(f"  features -> {features_path.name}")

        # Fit all models
        fits_df = fit_runner.run_all(features_df, verbose=False)
        out_path = config.RESULTS_DIR / f"fits_by_subject__{mode}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fits_df.to_parquet(out_path, index=False)
        print(f"  fits     -> {out_path.name}  ({len(fits_df)} rows)")
        print(f"  elapsed   {time.time() - t_mode:.1f} s\n")

    print(f"All modes complete in {time.time() - t0:.1f} s total.")


if __name__ == "__main__":
    main()
