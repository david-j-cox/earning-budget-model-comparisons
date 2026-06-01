"""Figures for the observed-vs-predicted comparison.

Outputs (under results/figures/):
  fig_calibration.png  : 18-panel grid; one panel per model. Each panel shows
                         binned predicted P(SS) (x) vs. observed choice rate (y),
                         colored by condition.
  fig_trajectory_all.png : 8-panel grid (4 conditions × 2 groups). Each panel
                         shows mean observed proportion-SS over trials and one
                         line per model. Per-family variants: fig_trajectory_<family>.png.

Uses scope='all' predictions (model fit pooled across conditions).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config


# Order conditions by budget sign (positive -> negative). Publication-safe
# grayscale: conditions distinguished by gray level + marker shape.
COND_ORDER = ["Chicken", "Crab", "Turtle", "Piranha"]
COND_COLORS = {
    "Chicken": "#000000",  # black             (greatest positive)
    "Crab":    "#555555",  # dark gray         (moderate positive)
    "Turtle":  "#888888",  # mid gray          (small negative)
    "Piranha": "#BBBBBB",  # light gray        (greatest negative)
}
COND_MARKERS = {
    "Chicken": "o",
    "Crab":    "s",
    "Turtle":  "^",
    "Piranha": "D",
}

MODEL_ORDER = [
    "random", "bias", "wsls", "logistic",
    "energy_budget", "energy_budget_threshold", "energy_budget_zscore", "energy_budget_categorical",
    "mvt", "mvt_fitted_rho",
    "melioration", "kinetic", "behavioral_momentum", "hill_climbing", "ratio_invariance",
    "q_learning", "q_dual_alpha", "q_forgetting", "q_dynamic_alpha", "q_condition_aware",
    "q_table_operant", "q_table_operant_budget",
    "human_as_qtable_operant", "human_as_qtable_operant_budget",
]

# Explicit programmed-contingency labels for each condition (replace animal names
# in panel titles). Values mirror Table 2 of the manuscript: net pt/s if SS-only.
COND_LABELS = {
    "Chicken": "+0.67 pt/s",
    "Crab":    "+0.50 pt/s",
    "Turtle":  "−0.05 pt/s",
    "Piranha": "−0.50 pt/s",
}

# Human-readable legend labels for models. Used wherever a model name would
# otherwise appear in a legend or figure annotation.
MODEL_LABELS = {
    "random":                         "Random",
    "bias":                           "Bias",
    "wsls":                           "WSLS",
    "logistic":                       "Logistic",
    "energy_budget":                  "Energy Budget\n(1-param Sigmoid)",
    "energy_budget_threshold":        "Energy Budget\n(2-param + Threshold)",
    "energy_budget_zscore":           "Energy Budget\n(Caraco z-score)",
    "energy_budget_categorical":      "Energy Budget\n(Categorical)",
    "mvt":                            "Marginal Value\n(Loss-Rate ρ)",
    "mvt_fitted_rho":                 "Marginal Value\n(Fitted ρ)",
    "melioration":                    "Melioration",
    "kinetic":                        "Kinetic",
    "behavioral_momentum":            "Behavioral Momentum",
    "hill_climbing":                  "Hill Climbing",
    "ratio_invariance":               "Ratio Invariance",
    "q_learning":                     "Q-Learning",
    "q_dual_alpha":                   "Q-Learning\n(Dual α)",
    "q_forgetting":                   "Q-Learning\n(Forgetting)",
    "q_dynamic_alpha":                "Q-Learning\n(Dynamic α)",
    "q_condition_aware":              "Q-Learning\n(Condition-Aware)",
    "q_table_operant":                "Q-Table\n(Operant State)",
    "q_table_operant_budget":         "Q-Table\n(Operant + Budget State)",
    "human_as_qtable_operant":        "Human-as-Q-Table\n(Operant State)",
    "human_as_qtable_operant_budget": "Human-as-Q-Table\n(Operant + Budget State)",
}


def pretty_model(name):
    return MODEL_LABELS.get(name, name.replace("_", " ").title())


# Shared styling constants
XYLABEL_FONTSIZE = 20
XYLABEL_PAD = 12
LEGEND_FONTSIZE = 14
PANEL_TITLE_FONTSIZE = 12
TICK_FONTSIZE = 14
HSPACE = 0.22
# Figure layout: panels occupy the left SUBPLOTS_RIGHT fraction of the figure;
# the legend sits just to the right of the rightmost panel.
SUBPLOTS_RIGHT = 0.80
LEGEND_ANCHOR = (0.805, 0.5)


def _despine(ax, sides=("top", "right", "bottom")):
    """Remove specified spines."""
    for side in sides:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=True, top=False)


def _panel_label(ax, text):
    """Place a panel descriptor in the lower-left corner inside the axes."""
    ax.text(0.02, 0.04, text, transform=ax.transAxes,
            fontsize=PANEL_TITLE_FONTSIZE, va="bottom", ha="left",
            color="black")

# Within-family styles for trajectory & residual plots. Each family pulls from
# this rotation so models in the same panel are visually distinct via
# (linestyle, marker) combos. No global uniqueness — only within a panel.
PANEL_STYLE_ROTATION = [
    {"color": "#000000", "ls": "-",          "marker": "",  "lw": 1.4, "ms": 4},
    {"color": "#444444", "ls": "--",         "marker": "",  "lw": 1.4, "ms": 4},
    {"color": "#666666", "ls": "-.",         "marker": "",  "lw": 1.4, "ms": 4},
    {"color": "#222222", "ls": (0, (1, 1)),  "marker": "",  "lw": 1.4, "ms": 4},
    {"color": "#888888", "ls": (0, (3, 1, 1, 1)), "marker": "", "lw": 1.4, "ms": 4},
    {"color": "#555555", "ls": (0, (5, 1)), "marker": "", "lw": 1.4, "ms": 4},
]


def style_for(model_index):
    return PANEL_STYLE_ROTATION[model_index % len(PANEL_STYLE_ROTATION)]


def calibration_figure(preds, out_path, scope="all", n_bins=10):
    """Per-model calibration: binned predicted P(SS) vs. observed choice rate."""
    df = preds[preds["scope"] == scope].copy()
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    n_models = len(models)
    n_cols = 4
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows),
                             squeeze=False)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for i, model in enumerate(models):
        ax = axes[i // n_cols, i % n_cols]
        mdf = df[df["model"] == model]
        for cond in COND_ORDER:
            sub = mdf[mdf["condition"] == cond]
            if len(sub) == 0:
                continue
            # Bin by predicted P(SS)
            p = sub["predicted_p_ss"].clip(0, 1).values
            o = sub["observed_choice"].astype(float).values
            bin_idx = np.digitize(p, bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            mean_obs = np.full(n_bins, np.nan)
            for b in range(n_bins):
                mask = bin_idx == b
                if mask.sum() >= 3:
                    mean_obs[b] = o[mask].mean()
            ax.plot(bin_centers, mean_obs, marker=COND_MARKERS[cond],
                    markersize=5, color=COND_COLORS[cond],
                    label=f"{cond} ({COND_LABELS[cond]})",
                    linewidth=1.2, markeredgecolor="black", markeredgewidth=0.5)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.5)
        ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=TICK_FONTSIZE - 4)
        # Calibration plot: keep the bottom axis since the diagonal anchors at 0.
        _despine(ax, sides=("top", "right"))
        # Panel label in lower-right corner
        ax.text(0.98, 0.04, pretty_model(model), transform=ax.transAxes,
                fontsize=PANEL_TITLE_FONTSIZE, va="bottom", ha="right",
                color="black")
        if i % n_cols == 0:
            ax.set_ylabel("Observed P(SS)", fontsize=XYLABEL_FONTSIZE,
                          labelpad=XYLABEL_PAD)
        if i // n_cols == n_rows - 1:
            ax.set_xlabel("Predicted P(SS)", fontsize=XYLABEL_FONTSIZE,
                          labelpad=XYLABEL_PAD)

    # Remove empty axes
    for j in range(n_models, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, SUBPLOTS_RIGHT, 1])
    fig.subplots_adjust(hspace=HSPACE, right=SUBPLOTS_RIGHT)
    fig.legend(handles, labels, loc="center left", fontsize=LEGEND_FONTSIZE,
               frameon=False, bbox_to_anchor=LEGEND_ANCHOR,
               bbox_transform=fig.transFigure)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap 95% CI on the mean of a binary 0/1 vector."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (np.nan, np.nan)
    if n == 1:
        return (float(values[0]), float(values[0]))
    boots = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def trajectory_figure(preds, features, out_path, scope="all", max_trial=60,
                      min_count=5):
    """For each (condition, group), plot mean observed P(SS) over trials with
    bootstrap 95% CI markers, overlaid with predicted P(SS) lines from each model.
    Each model gets a distinct (color, linestyle) combination via TRAJ_STYLE.
    """
    df = preds[preds["scope"] == scope].copy()
    df = df[df["trial"] <= max_trial]
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), squeeze=False,
                             sharex=True, sharey=True)
    for r, grp in enumerate(["05", "10"]):
        for c, cond in enumerate(COND_ORDER):
            ax = axes[r, c]
            sub = df[(df["group"] == grp) & (df["condition"] == cond)]
            if len(sub) == 0:
                ax.set_title(f"Group {grp} / {cond} (no data)")
                continue

            # Model lines first (so observed markers sit on top)
            for mi, m in enumerate(models):
                mdf = sub[sub["model"] == m]
                if len(mdf) == 0:
                    continue
                pred = (mdf.groupby("trial")["predicted_p_ss"]
                        .agg(["mean", "count"]).reset_index())
                pred = pred[pred["count"] >= min_count]
                style = style_for(mi)
                ax.plot(pred["trial"], pred["mean"],
                        color=style["color"], linestyle=style["ls"],
                        linewidth=style["lw"], alpha=0.9,
                        label=pretty_model(m))

            # Observed: mean ± bootstrap 95% CI, with markers
            trials = sorted(sub["trial"].unique())
            xs, ys, los, his = [], [], [], []
            for t in trials:
                vals = sub[sub["trial"] == t]["observed_choice"].astype(float).values
                # Trim to one row per subject (across models all rows for a
                # (subject,trial) share the same observed_choice).
                vals = sub[sub["trial"] == t].drop_duplicates(
                    subset=["subject"])["observed_choice"].astype(float).values
                if len(vals) < min_count:
                    continue
                m = float(vals.mean())
                lo, hi = _bootstrap_ci(vals)
                xs.append(t); ys.append(m); los.append(lo); his.append(hi)
            if xs:
                xs = np.asarray(xs); ys = np.asarray(ys)
                los = np.asarray(los); his = np.asarray(his)
                ax.errorbar(xs, ys, yerr=[ys - los, his - ys],
                            fmt="o", color="black", markersize=4,
                            elinewidth=0.8, capsize=2, alpha=0.2, zorder=5,
                            label="Observed (Mean ± 95% CI)")

            ax.set_ylim(-0.02, 1.02)
            ax.axhline(0.5, color="grey", linewidth=0.4, linestyle=":")
            ax.tick_params(labelsize=TICK_FONTSIZE)
            _despine(ax)
            _panel_label(ax, f"Group {grp}: {COND_LABELS[cond]}")
            if r == 1:
                ax.set_xlabel("Trial within condition",
                              fontsize=XYLABEL_FONTSIZE, labelpad=XYLABEL_PAD)
            if c == 0:
                ax.set_ylabel("P(choose SS)",
                              fontsize=XYLABEL_FONTSIZE, labelpad=XYLABEL_PAD)
    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if "Observed (Mean ± 95% CI)" in labels:
        idx = labels.index("Observed (Mean ± 95% CI)")
        order = [idx] + [i for i in range(len(labels)) if i != idx]
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]
    fig.tight_layout(rect=[0, 0, SUBPLOTS_RIGHT, 1])
    fig.subplots_adjust(hspace=HSPACE, right=SUBPLOTS_RIGHT)
    fig.legend(handles, labels, loc="center left", fontsize=LEGEND_FONTSIZE,
               frameon=False, bbox_to_anchor=LEGEND_ANCHOR,
               bbox_transform=fig.transFigure)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


FAMILY_MODELS = {
    "baseline":            ["random", "bias", "wsls", "logistic"],
    "energy_budget":       ["energy_budget", "energy_budget_threshold",
                            "energy_budget_zscore", "energy_budget_categorical",
                            "mvt", "mvt_fitted_rho"],
    "historical_dynamics": ["melioration", "kinetic", "behavioral_momentum",
                            "hill_climbing", "ratio_invariance"],
    "rl_action_value":     ["q_learning", "q_dual_alpha", "q_forgetting",
                            "q_dynamic_alpha", "q_condition_aware"],
    "rl_state_action":     ["q_table_operant", "q_table_operant_budget"],
    "human_as_qtable":     ["human_as_qtable_operant", "human_as_qtable_operant_budget"],
}


def trajectory_by_family_figure(preds, family, models, out_path,
                                 scope="all", max_trial=60, min_count=5):
    """Trajectory plot restricted to one model family. Each panel is one
    (group, condition); each family's models overlaid in distinct styles."""
    df = preds[(preds["scope"] == scope) & (preds["model"].isin(models))].copy()
    df = df[df["trial"] <= max_trial]
    if len(df) == 0:
        return None

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), squeeze=False,
                             sharex=True, sharey=True)
    for r, grp in enumerate(["05", "10"]):
        for c, cond in enumerate(COND_ORDER):
            ax = axes[r, c]
            sub = df[(df["group"] == grp) & (df["condition"] == cond)]
            # All models in family — distinct styles within this panel
            for mi, m in enumerate(models):
                mdf = sub[sub["model"] == m]
                if len(mdf) == 0:
                    continue
                pred = (mdf.groupby("trial")["predicted_p_ss"]
                        .agg(["mean", "count"]).reset_index())
                pred = pred[pred["count"] >= min_count]
                style = style_for(mi)
                ax.plot(pred["trial"], pred["mean"],
                        color=style["color"], linestyle=style["ls"],
                        linewidth=style["lw"], alpha=0.95,
                        label=pretty_model(m))

            # Observed mean + 95% CI markers
            sub_all = preds[(preds["scope"] == scope) &
                            (preds["group"] == grp) &
                            (preds["condition"] == cond) &
                            (preds["trial"] <= max_trial)]
            trials = sorted(sub_all["trial"].unique())
            xs, ys, los, his = [], [], [], []
            for t in trials:
                vals = sub_all[sub_all["trial"] == t].drop_duplicates(
                    subset=["subject"])["observed_choice"].astype(float).values
                if len(vals) < min_count:
                    continue
                m_val = float(vals.mean()); lo, hi = _bootstrap_ci(vals)
                xs.append(t); ys.append(m_val); los.append(lo); his.append(hi)
            if xs:
                xs = np.asarray(xs); ys = np.asarray(ys)
                los = np.asarray(los); his = np.asarray(his)
                ax.errorbar(xs, ys, yerr=[ys - los, his - ys],
                            fmt="o", color="black", markersize=4,
                            elinewidth=0.8, capsize=2, alpha=0.2, zorder=5,
                            label="Observed (Mean ± 95% CI)")

            ax.set_ylim(-0.02, 1.02)
            ax.axhline(0.5, color="grey", linewidth=0.4, linestyle=":")
            ax.tick_params(labelsize=TICK_FONTSIZE)
            _despine(ax)
            _panel_label(ax, f"Group {grp}: {COND_LABELS[cond]}")
            if r == 1:
                ax.set_xlabel("Trial within condition",
                              fontsize=XYLABEL_FONTSIZE, labelpad=XYLABEL_PAD)
            if c == 0:
                ax.set_ylabel("P(choose SS)",
                              fontsize=XYLABEL_FONTSIZE, labelpad=XYLABEL_PAD)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if "Observed (Mean ± 95% CI)" in labels:
        idx = labels.index("Observed (Mean ± 95% CI)")
        order = [idx] + [i for i in range(len(labels)) if i != idx]
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]
    fig.tight_layout(rect=[0, 0, SUBPLOTS_RIGHT, 1])
    fig.subplots_adjust(hspace=HSPACE, right=SUBPLOTS_RIGHT)
    fig.legend(handles, labels, loc="center left", fontsize=LEGEND_FONTSIZE,
               frameon=False, bbox_to_anchor=LEGEND_ANCHOR,
               bbox_transform=fig.transFigure)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def residual_by_family_figure(preds, family, models, out_path,
                               scope="all", max_trial=60, min_count=5):
    """Per-family residual plot: (observed P_SS) - (predicted P_SS) over trials,
    one panel per (group, condition). Zero line dashed; residuals above zero mean
    the model under-predicts SS, below zero means over-predicts."""
    df = preds[(preds["scope"] == scope) & (preds["model"].isin(models))].copy()
    df = df[df["trial"] <= max_trial]
    if len(df) == 0:
        return None

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), squeeze=False,
                             sharex=True, sharey=True)
    for r, grp in enumerate(["05", "10"]):
        for c, cond in enumerate(COND_ORDER):
            ax = axes[r, c]
            sub = df[(df["group"] == grp) & (df["condition"] == cond)]
            # Mean observed P(SS) per trial across subjects, deduped on subject
            obs_unique = (preds[(preds["scope"] == scope) &
                                (preds["group"] == grp) &
                                (preds["condition"] == cond) &
                                (preds["trial"] <= max_trial)]
                          .drop_duplicates(subset=["subject", "trial"])
                          [["trial", "observed_choice"]])
            obs_mean = (obs_unique.groupby("trial")["observed_choice"]
                        .agg(["mean", "count"]).reset_index())
            obs_mean = obs_mean[obs_mean["count"] >= min_count]
            obs_lookup = dict(zip(obs_mean["trial"], obs_mean["mean"]))

            for mi, m in enumerate(models):
                mdf = sub[sub["model"] == m]
                if len(mdf) == 0:
                    continue
                pred = (mdf.groupby("trial")["predicted_p_ss"]
                        .agg(["mean", "count"]).reset_index())
                pred = pred[pred["count"] >= min_count]
                # Match to observed
                pred["obs"] = pred["trial"].map(obs_lookup)
                pred = pred.dropna(subset=["obs"])
                pred["residual"] = pred["obs"] - pred["mean"]
                style = style_for(mi)
                ax.plot(pred["trial"], pred["residual"],
                        color=style["color"], linestyle=style["ls"],
                        linewidth=style["lw"], alpha=0.95,
                        label=pretty_model(m))

            ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
            ax.set_ylim(-1.02, 1.02)
            ax.tick_params(labelsize=TICK_FONTSIZE)
            _despine(ax)
            _panel_label(ax, f"Group {grp}: {COND_LABELS[cond]}")
            if r == 1:
                ax.set_xlabel("Trial within condition",
                              fontsize=XYLABEL_FONTSIZE, labelpad=XYLABEL_PAD)
            if c == 0:
                ax.set_ylabel("Observed − Predicted P(SS)",
                              fontsize=XYLABEL_FONTSIZE, labelpad=XYLABEL_PAD)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, SUBPLOTS_RIGHT, 1])
    fig.subplots_adjust(hspace=HSPACE, right=SUBPLOTS_RIGHT)
    fig.legend(handles, labels, loc="center left", fontsize=LEGEND_FONTSIZE,
               frameon=False, bbox_to_anchor=LEGEND_ANCHOR,
               bbox_transform=fig.transFigure)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def combined_by_family_figure(preds, family, models, out_path,
                              scope="all", max_trial=60, min_count=5):
    """One publication figure per model family on a 4x4 grid.

    Rows are the four conditions (positive -> negative budget). Columns are:
      col 0: model fit (observed P(SS) + model lines), Group 5:1
      col 1: residual (observed - predicted P(SS)),     Group 5:1
      col 2: model fit,                                  Group 10:1
      col 3: residual,                                   Group 10:1
    A single horizontal legend sits above the grid in the title area. The
    line/marker/label/legend sizes below were tuned for the manuscript figures.
    """
    df = preds[(preds["scope"] == scope) & (preds["model"].isin(models))].copy()
    df = df[df["trial"] <= max_trial]
    if len(df) == 0:
        return None

    col_spec = [("05", "fit"), ("05", "resid"), ("10", "fit"), ("10", "resid")]
    col_headers = ["Group 5:1\nModel Fit", "Group 5:1\nResidual",
                   "Group 10:1\nModel Fit", "Group 10:1\nResidual"]

    fig, axes = plt.subplots(4, 4, figsize=(20, 18), squeeze=False, sharex=True)

    for r, cond in enumerate(COND_ORDER):
        for c, (grp, kind) in enumerate(col_spec):
            ax = axes[r, c]
            sub = df[(df["group"] == grp) & (df["condition"] == cond)]

            obs_unique = (preds[(preds["scope"] == scope) &
                                (preds["group"] == grp) &
                                (preds["condition"] == cond) &
                                (preds["trial"] <= max_trial)]
                          .drop_duplicates(subset=["subject", "trial"])
                          [["subject", "trial", "observed_choice"]])
            obs_mean = (obs_unique.groupby("trial")["observed_choice"]
                        .agg(["mean", "count"]).reset_index())
            obs_mean = obs_mean[obs_mean["count"] >= min_count]
            obs_lookup = dict(zip(obs_mean["trial"], obs_mean["mean"]))

            for mi, m in enumerate(models):
                mdf = sub[sub["model"] == m]
                if len(mdf) == 0:
                    continue
                pred = (mdf.groupby("trial")["predicted_p_ss"]
                        .agg(["mean", "count"]).reset_index())
                pred = pred[pred["count"] >= min_count]
                style = style_for(mi)
                if kind == "fit":
                    ax.plot(pred["trial"], pred["mean"],
                            color=style["color"], linestyle=style["ls"],
                            linewidth=2.2, alpha=0.95, label=pretty_model(m))
                else:
                    pred["obs"] = pred["trial"].map(obs_lookup)
                    pred = pred.dropna(subset=["obs"])
                    pred["residual"] = pred["obs"] - pred["mean"]
                    ax.plot(pred["trial"], pred["residual"],
                            color=style["color"], linestyle=style["ls"],
                            linewidth=2.2, alpha=0.95, label=pretty_model(m))

            if kind == "fit":
                # Observed mean + bootstrap 95% CI markers on top of model lines.
                xs, ys, los, his = [], [], [], []
                for t in sorted(obs_unique["trial"].unique()):
                    vals = (obs_unique[obs_unique["trial"] == t]
                            ["observed_choice"].astype(float).values)
                    if len(vals) < min_count:
                        continue
                    lo, hi = _bootstrap_ci(vals)
                    xs.append(t); ys.append(float(vals.mean()))
                    los.append(lo); his.append(hi)
                if xs:
                    xs = np.asarray(xs); ys = np.asarray(ys)
                    los = np.asarray(los); his = np.asarray(his)
                    ax.errorbar(xs, ys, yerr=[ys - los, his - ys],
                                fmt="o", color="black", markersize=7,
                                elinewidth=0.8, capsize=2, alpha=0.2, zorder=5,
                                label="Observed (Mean ± 95% CI)")
                ax.set_ylim(-0.02, 1.02)
                ax.axhline(0.5, color="grey", linewidth=0.4, linestyle=":")
            else:
                ax.axhline(0, color="black", linewidth=0.7, linestyle="--",
                           alpha=0.5)
                ax.set_ylim(-1.02, 1.02)

            ax.tick_params(labelsize=TICK_FONTSIZE)
            _despine(ax)

            if r == 0:
                ax.set_title(col_headers[c], fontsize=XYLABEL_FONTSIZE,
                             fontweight="bold", pad=14)
            if c == 0:
                ax.set_ylabel(f"{cond}\n({COND_LABELS[cond]})",
                              fontsize=24, labelpad=XYLABEL_PAD,
                              fontweight="bold")

    # Single horizontal legend above the grid; pull handles from a fit panel so
    # the Observed series is included, ordered first.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if "Observed (Mean ± 95% CI)" in labels:
        idx = labels.index("Observed (Mean ± 95% CI)")
        order = [idx] + [i for i in range(len(labels)) if i != idx]
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]

    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    fig.subplots_adjust(hspace=HSPACE, wspace=0.18)
    # Shared x-axis label across all columns (2x the former per-panel size),
    # placed below the bottom-row ticks.
    fig.supxlabel("Trial within condition", fontsize=(XYLABEL_FONTSIZE - 2) * 2,
                  y=0.015)
    fig.legend(handles, labels, loc="upper center", fontsize=20, frameon=False,
               ncol=len(labels), markerscale=1.75, bbox_to_anchor=(0.5, 0.99),
               bbox_transform=fig.transFigure, columnspacing=1.6,
               handlelength=2.6)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# Distinct (gray level, linestyle) combinations for the best-per-family model
# lines in a single-participant panel. Solid black is reserved for the observed
# rolling mean, so every model style here is non-solid to keep it visually
# separable from the observed series. Publication-safe grayscale.
FAMILY_LINE_STYLES = [
    {"color": "#222222", "ls": "--"},
    {"color": "#555555", "ls": "-."},
    {"color": "#777777", "ls": ":"},
    {"color": "#333333", "ls": (0, (3, 1, 1, 1))},
    {"color": "#888888", "ls": (0, (5, 1))},
    {"color": "#666666", "ls": (0, (1, 1))},
    {"color": "#000000", "ls": (0, (3, 1, 1, 1, 1, 1))},
]


def best_model_per_family(fits, scope="all"):
    """Return {model_family: model} for the lowest-median-AICc model in each
    family at the pooled (condition='all') fit scope. Families are returned in
    FAMILY_MODELS order so the legend is stable across participants."""
    d = fits[(fits["scope"] == scope) & (fits["condition"] == "all")]
    med = d.groupby(["model_family", "model"])["aicc"].median().reset_index()
    best = {}
    for fam in FAMILY_MODELS:
        sub = med[med["model_family"] == fam].sort_values("aicc")
        if len(sub):
            best[fam] = sub.iloc[0]["model"]
    return best


def participant_figure(preds, subject, models, out_path, scope="all"):
    """Single-participant fit figure: a 2x2 grid (one panel per condition).

    Each panel shows the participant's raw trial-by-trial choices (black x
    marks, offset just outside the [0, 1] band) and the predicted P(SS)
    trajectory of each supplied model (best-per-family). x-axes are independent
    because trials-per-condition vary widely across participants.
    """
    sdf = preds[(preds["scope"] == scope) & (preds["subject"] == subject)].copy()
    if len(sdf) == 0:
        return None
    grp = sdf["group"].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), squeeze=False, sharey=True)
    for idx, cond in enumerate(COND_ORDER):
        ax = axes[idx // 2, idx % 2]
        cdf = sdf[sdf["condition"] == cond]
        ax.set_title(f"{cond} ({COND_LABELS[cond]})",
                     fontsize=PANEL_TITLE_FONTSIZE, pad=8)
        if len(cdf) == 0:
            ax.text(0.5, 0.5, "no trials", transform=ax.transAxes,
                    ha="center", va="center", color="grey",
                    fontsize=PANEL_TITLE_FONTSIZE)
            ax.set_ylim(-0.18, 1.18)
            _despine(ax)
            continue

        # Model prediction lines (drawn first so observed sits on top).
        for mi, m in enumerate(models):
            mdf = cdf[cdf["model"] == m].sort_values("trial")
            if len(mdf) == 0:
                continue
            st = FAMILY_LINE_STYLES[mi % len(FAMILY_LINE_STYLES)]
            ax.plot(mdf["trial"], mdf["predicted_p_ss"],
                    color=st["color"], linestyle=st["ls"], linewidth=1.6,
                    alpha=0.95, label=pretty_model(m))

        # Observed choices are identical across models for a (subject, trial),
        # so dedupe on trial before plotting. Raw markers are offset just
        # outside the [0, 1] band (LL=0 -> -0.1, SS=1 -> 1.1) so they don't
        # collide with the model P(SS) lines.
        obs = (cdf.drop_duplicates(subset=["trial"]).sort_values("trial"))
        obs_y = obs["observed_choice"].astype(float)
        ax.plot(obs["trial"], -0.1 + 1.2 * obs_y,
                linestyle="", marker="x", markersize=4, color="black",
                markeredgewidth=0.9,
                label="Observed choice (SS=1, LL=0)", zorder=6)

        ax.set_ylim(-0.18, 1.18)
        ax.set_yticks([0, 0.5, 1.0])
        ax.axhline(0.5, color="grey", linewidth=0.4, linestyle=":")
        ax.tick_params(labelsize=TICK_FONTSIZE)
        _despine(ax)
        if idx // 2 == 1:
            ax.set_xlabel("Trial Within Condition",
                          fontsize=XYLABEL_FONTSIZE, labelpad=XYLABEL_PAD)
        if idx % 2 == 0:
            ax.set_ylabel("P(choose SS)",
                          fontsize=XYLABEL_FONTSIZE, labelpad=XYLABEL_PAD)

    # Collect legend handles from whichever panel has the most series.
    handles, labels = [], []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        if len(l) > len(labels):
            handles, labels = h, l
    # Put the two observed entries first.
    obs_first = [i for i, l in enumerate(labels) if l.startswith("Observed")]
    rest = [i for i in range(len(labels)) if i not in obs_first]
    order = obs_first + rest
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]

    grp_ratio = {"05": "5:1", "10": "10:1"}.get(grp, grp)
    pid = subject.split("_")[-1]  # "Exp1_05_P15" -> "P15"
    fig.suptitle(f"Participant {pid} (Group {grp_ratio})",
                 fontsize=XYLABEL_FONTSIZE + 4, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.03, 1, 0.86])
    fig.subplots_adjust(hspace=0.4)
    # Legend sits between the suptitle and the top row of subplots.
    fig.legend(handles, labels, loc="upper center", fontsize=LEGEND_FONTSIZE,
               frameon=False, ncol=4, bbox_to_anchor=(0.5, 0.93),
               bbox_transform=fig.transFigure, columnspacing=1.4)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main():
    preds = pd.read_parquet(config.RESULTS_DIR / "predictions.parquet")
    features = pd.read_parquet(config.TRANSFORMED_DIR / "exp1_features.parquet")
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)

    p1 = calibration_figure(preds, config.FIG_DIR / "fig_calibration.png")
    print(f"  Wrote {p1}")
    p2 = trajectory_figure(preds, features, config.FIG_DIR / "fig_trajectory_all.png")
    print(f"  Wrote {p2}")
    # One trajectory + residual figure per family
    for fam, models in FAMILY_MODELS.items():
        p = trajectory_by_family_figure(
            preds, fam, models,
            config.FIG_DIR / f"fig_trajectory_{fam}.png")
        if p is not None:
            print(f"  Wrote {p}")
        p = residual_by_family_figure(
            preds, fam, models,
            config.FIG_DIR / f"fig_residual_{fam}.png")
        if p is not None:
            print(f"  Wrote {p}")
        # Combined fit + residual grid (rows = conditions, cols = group x kind)
        p = combined_by_family_figure(
            preds, fam, models,
            config.FIG_DIR / f"fig_combined_{fam}.png")
        if p is not None:
            print(f"  Wrote {p}")


if __name__ == "__main__":
    main()
