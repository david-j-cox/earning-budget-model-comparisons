"""Stage 1 of the pipeline: load raw Exp1 data, engineer features, fit all models.

Usage from the repo root:
    python scripts/run.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import time
import pandas as pd

from pipeline import config, load_data, feature_engineering, fit_runner


def main():
    t0 = time.time()

    # 1. Load
    print("=" * 60)
    print("Step 1: Load Exp1 raw data")
    print("=" * 60)
    long_df = load_data.load_exp1(verbose=True)
    load_data.save_long(long_df)

    # 2. Feature engineering
    print("\n" + "=" * 60)
    print("Step 2: Feature engineering")
    print("=" * 60)
    features_df = feature_engineering.build_features(long_df)
    features_path = config.TRANSFORMED_DIR / "exp1_features.parquet"
    features_df.to_parquet(features_path, index=False)
    print(f"  Saved {len(features_df)} feature rows to {features_path.name}")

    # 3. Fit all models per subject
    print("\n" + "=" * 60)
    print("Step 3: Fit models")
    print("=" * 60)
    fits_df = fit_runner.run_all(features_df)
    out_path = config.RESULTS_DIR / "fits_by_subject.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits_df.to_parquet(out_path, index=False)
    print(f"\n  Saved {len(fits_df)} fit rows to {out_path}")

    # 4. Summary
    print("\n" + "=" * 60)
    print("Step 4: Summary")
    print("=" * 60)
    print(f"\nN subjects:   {fits_df['subject'].nunique()}")
    print(f"N models:     {fits_df['model'].nunique()}")
    print(f"N fit rows:   {len(fits_df)}")
    print(f"\nFamilies × scopes:")
    print(fits_df.groupby(["model_family", "scope"]).size().unstack(fill_value=0))

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
