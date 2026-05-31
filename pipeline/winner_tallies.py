"""Per-subject winner tallies.

For each subject (pooled-scope) and each subject × condition (per-condition scope),
identify the model with the lowest AICc. Tally winners per group and per
(group, condition). Outputs:

  results/winners_pooled.parquet
  results/winners_by_condition.parquet
  results/figures/fig_winners_pooled.png
  results/figures/fig_winners_by_condition.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config
from .make_figures import (
    pretty_model, MODEL_ORDER, COND_ORDER, COND_LABELS,
    XYLABEL_FONTSIZE, XYLABEL_PAD, LEGEND_FONTSIZE,
    TICK_FONTSIZE, HSPACE, SUBPLOTS_RIGHT, _despine,
)


def find_winners(fits, scope, by_condition=False):
    """For each (subject [, condition]) row, return the model with lowest AICc.

    Subjects with all-NaN AICc in their group (rare) are excluded with a count.
    """
    sub = fits[fits["scope"] == scope].copy()
    sub = sub.dropna(subset=["aicc"])
    keys = ["subject", "group"]
    if by_condition:
        keys = keys + ["condition"]
    # idxmin per group
    idx = sub.groupby(keys)["aicc"].idxmin()
    winners = sub.loc[idx, keys + ["model", "model_family", "aicc"]].reset_index(drop=True)
    return winners


def tally(winners, by_condition=False):
    """Count wins per model, optionally per condition. Returns long-format df."""
    keys = ["group", "model"]
    if by_condition:
        keys = ["group", "condition", "model"]
    counts = winners.groupby(keys).size().reset_index(name="n_wins")
    totals = winners.groupby([k for k in keys if k != "model"]).size().reset_index(
        name="n_subjects")
    df = counts.merge(totals, on=[k for k in keys if k != "model"])
    df["pct_wins"] = 100 * df["n_wins"] / df["n_subjects"]
    return df


def pooled_figure(tally_df, out_path):
    """Stacked horizontal bar: % subject-wins per model, split by group."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    models = [m for m in MODEL_ORDER if m in tally_df["model"].unique()]
    for ax, grp in zip(axes, ["05", "10"]):
        df = tally_df[tally_df["group"] == grp].set_index("model").reindex(models)
        counts = df["n_wins"].fillna(0).values
        pcts = df["pct_wins"].fillna(0).values
        n_subj = int(df["n_subjects"].dropna().iloc[0]) if df["n_subjects"].dropna().size else 0
        labels = [pretty_model(m) for m in models]
        # Sort by count
        order = np.argsort(-counts)
        ax.barh(range(len(models)), counts[order], color="#444444",
                edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([labels[i] for i in order], fontsize=11)
        ax.set_xlabel("Subjects won (count)", fontsize=XYLABEL_FONTSIZE,
                      labelpad=XYLABEL_PAD)
        # Annotate counts and percentages
        for i, c in enumerate(counts[order]):
            if c > 0:
                ax.text(c + 0.5, i, f"{int(c)} ({pcts[order][i]:.0f}%)",
                        fontsize=10, va="center")
        _despine(ax)
        ax.text(0.98, 0.04, f"Group {grp} (N = {n_subj})", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=14)
        ax.invert_yaxis()
        ax.set_xlim(0, max(counts.max() * 1.18, 5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def by_condition_figure(tally_df, out_path):
    """4-column × 2-row grid: bar per model, one panel per (group, condition)."""
    models = [m for m in MODEL_ORDER if m in tally_df["model"].unique()]
    fig, axes = plt.subplots(2, 4, figsize=(20, 12), sharey=True, squeeze=False)
    for r, grp in enumerate(["05", "10"]):
        for c, cond in enumerate(COND_ORDER):
            ax = axes[r, c]
            df = (tally_df[(tally_df["group"] == grp) & (tally_df["condition"] == cond)]
                  .set_index("model").reindex(models))
            counts = df["n_wins"].fillna(0).values
            pcts = df["pct_wins"].fillna(0).values
            n_subj = int(df["n_subjects"].dropna().iloc[0]) if df["n_subjects"].dropna().size else 0
            order = np.argsort(-counts)
            labels = [pretty_model(m) for m in models]
            ax.barh(range(len(models)), counts[order], color="#444444",
                    edgecolor="black", linewidth=0.5)
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels([labels[i] for i in order], fontsize=8)
            ax.invert_yaxis()
            ax.tick_params(labelsize=TICK_FONTSIZE - 2)
            _despine(ax)
            ax.text(0.98, 0.04,
                    f"Group {grp}: {COND_LABELS[cond]}\n(N = {n_subj})",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=11)
            for i, cc in enumerate(counts[order]):
                if cc > 0:
                    ax.text(cc + 0.3, i, f"{int(cc)}",
                            fontsize=8, va="center")
            if r == 1:
                ax.set_xlabel("Wins", fontsize=XYLABEL_FONTSIZE,
                              labelpad=XYLABEL_PAD)
            ax.set_xlim(0, max(counts.max() * 1.18, 3))
    fig.tight_layout()
    fig.subplots_adjust(hspace=HSPACE, wspace=0.4)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    fits = pd.read_parquet(config.RESULTS_DIR / "fits_by_subject.parquet")

    # Pooled winners
    winners_pooled = find_winners(fits, scope="all", by_condition=False)
    tally_pooled = tally(winners_pooled, by_condition=False)
    out_p = config.RESULTS_DIR / "winners_pooled.parquet"
    tally_pooled.to_parquet(out_p, index=False)
    print(f"  Wrote {out_p}")

    fig_p = config.FIG_DIR / "fig_winners_pooled.png"
    pooled_figure(tally_pooled, fig_p)
    print(f"  Wrote {fig_p}")

    # By-condition winners
    winners_cond = find_winners(fits, scope="condition", by_condition=True)
    tally_cond = tally(winners_cond, by_condition=True)
    out_c = config.RESULTS_DIR / "winners_by_condition.parquet"
    tally_cond.to_parquet(out_c, index=False)
    print(f"  Wrote {out_c}")

    fig_c = config.FIG_DIR / "fig_winners_by_condition.png"
    by_condition_figure(tally_cond, fig_c)
    print(f"  Wrote {fig_c}")

    # Print summary
    print("\n=== Pooled (scope='all') ===")
    print(tally_pooled.sort_values(["group", "n_wins"], ascending=[True, False])
          .to_string(index=False))


if __name__ == "__main__":
    main()
