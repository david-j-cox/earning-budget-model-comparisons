"""Full pairwise paired-Wilcoxon model comparisons on per-subject ΔAICc.

For each pair (model_i, model_j), within each group and scope=='all',
test whether per-subject AICc differs significantly.

Outputs:
  results/pairwise_wilcoxon.parquet : long-format results
  results/figures/fig_pairwise_heatmap_grp05.png
  results/figures/fig_pairwise_heatmap_grp10.png

Effect size: median ΔAICc (model_i − model_j). Negative = model_i has better fit.
Significance: paired Wilcoxon signed-rank, Bonferroni-corrected for n×(n-1)/2 tests.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

from . import config


def pairwise_wilcoxon(fits, scope="all", metric="aicc"):
    """Within each group, compute pairwise paired-Wilcoxon on `metric` across subjects."""
    fits = fits[fits["scope"] == scope]
    rows = []
    for grp in fits["group"].unique():
        gfits = fits[fits["group"] == grp]
        # Pivot to subject × model
        wide = gfits.pivot_table(index="subject", columns="model", values=metric)
        models = list(wide.columns)
        n_tests = len(models) * (len(models) - 1) / 2
        bonf = max(n_tests, 1)
        for i, mi in enumerate(models):
            for j, mj in enumerate(models):
                if j <= i:
                    continue
                paired = wide[[mi, mj]].dropna()
                if len(paired) < 5:
                    rows.append(dict(group=grp, model_i=mi, model_j=mj,
                                     n=len(paired),
                                     median_diff=np.nan, W=np.nan, p=np.nan,
                                     p_bonf=np.nan, sig=False))
                    continue
                diffs = paired[mi].values - paired[mj].values
                median_diff = float(np.median(diffs))
                try:
                    stat, pval = wilcoxon(diffs, zero_method="zsplit")
                except Exception:
                    stat, pval = np.nan, np.nan
                p_bonf = min(pval * bonf, 1.0) if np.isfinite(pval) else np.nan
                rows.append(dict(
                    group=grp, model_i=mi, model_j=mj, n=len(paired),
                    median_diff=median_diff, W=float(stat) if np.isfinite(stat) else np.nan,
                    p=float(pval) if np.isfinite(pval) else np.nan,
                    p_bonf=p_bonf, sig=bool(p_bonf < 0.05) if np.isfinite(p_bonf) else False,
                ))
    return pd.DataFrame(rows)


def heatmap(pw_df, group, out_path, metric_name="ΔAICc"):
    """One heatmap per group: lower-triangle = median diff, upper-triangle = sig stars."""
    sub = pw_df[pw_df["group"] == group]
    models = sorted(set(sub["model_i"]).union(set(sub["model_j"])))
    # Use the actual order from the canonical list
    from .make_figures import MODEL_ORDER
    models = [m for m in MODEL_ORDER if m in models]
    n = len(models)

    diff_mat = np.full((n, n), np.nan)
    sig_mat = np.zeros((n, n), dtype=int)  # 0=ns, 1=p<.05, 2=p<.01, 3=p<.001 (bonferroni)
    idx = {m: i for i, m in enumerate(models)}

    for _, r in sub.iterrows():
        i = idx[r["model_i"]]; j = idx[r["model_j"]]
        # Order so lower-triangle has i > j
        a, b = (i, j) if i < j else (j, i)
        # Place median diff in lower triangle (row > col), so use (b, a)
        # Convention: row model − col model
        diff_mat[b, a] = r["median_diff"] if i < j else -r["median_diff"]
        diff_mat[a, b] = -diff_mat[b, a]
        if r["sig"]:
            p_bonf = r["p_bonf"]
            stars = (3 if p_bonf < 0.001 else 2 if p_bonf < 0.01 else 1)
        else:
            stars = 0
        sig_mat[b, a] = stars
        sig_mat[a, b] = stars

    fig, ax = plt.subplots(figsize=(0.55 * n + 3, 0.55 * n + 2))
    vmax = float(np.nanpercentile(np.abs(diff_mat), 95)) or 1.0
    im = ax.imshow(diff_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(n)); ax.set_xticklabels(models, rotation=90, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(models, fontsize=8)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = diff_mat[i, j]
            if not np.isfinite(v):
                continue
            star = "*" * sig_mat[i, j]
            txt = f"{v:.0f}\n{star}" if star else f"{v:.0f}"
            color = "white" if abs(v) > vmax * 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6, color=color)
    ax.set_title(f"Group {group}: median (row − col) {metric_name}, paired Wilcoxon (Bonferroni-corrected)\n"
                 f"Negative = row model has better fit. Stars: * p<.05, ** p<.01, *** p<.001", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    fits = pd.read_parquet(config.RESULTS_DIR / "fits_by_subject.parquet")
    pw = pairwise_wilcoxon(fits, scope="all", metric="aicc")
    out = config.RESULTS_DIR / "pairwise_wilcoxon.parquet"
    pw.to_parquet(out, index=False)
    print(f"  Wrote {out}")
    for grp in pw["group"].unique():
        p = heatmap(pw, grp, config.FIG_DIR / f"fig_pairwise_heatmap_grp{grp}.png")
        print(f"  Wrote {p}")

    # Summary
    n_sig = int(pw["sig"].sum())
    print(f"\nTotal pairwise tests: {len(pw)}")
    print(f"Significant after Bonferroni: {n_sig}")
    print(f"\nTop 15 strongest comparisons by |median ΔAICc| (group 05):")
    print(pw[pw['group']=='05'].sort_values('median_diff', key=lambda s: -s.abs())
          .head(15)[['model_i','model_j','n','median_diff','p_bonf','sig']].to_string(index=False))


if __name__ == "__main__":
    main()
