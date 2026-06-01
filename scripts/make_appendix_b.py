"""Appendix B: individual-participant model-fit figures.

Selects a random, group-balanced subset of participants (default 4 from each of
the 5:1 and 10:1 groups) restricted to those with at least 15 recorded choices
and no exclusive preference for one alternative in every condition (Piranha
exempt in the 5:1 group, where the condition ends before 15 choices accrue),
and writes one figure per participant under
results/figures/appendix_b/. Each figure is a 2x2 grid (one panel per condition)
overlaying the participant's observed SS choices (raw markers)
with the predicted P(SS) of the best-performing model in each family (lowest
median AICc at the pooled fit scope). A companion Appendix_B.tex that includes
the generated PNGs is written to the repo root.

The participant draw is seeded (default 42) so the appendix is reproducible.
Run from the repo root:  python scripts/make_appendix_b.py
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # repo root, for `import pipeline`

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.make_figures import (
    best_model_per_family, participant_figure, pretty_model, FAMILY_MODELS,
)

N_PER_GROUP = 4
SEED = 42

# Eligibility: a participant is drawable only if they have at least MIN_CHOICES
# recorded choices in every condition except those exempted for their group.
# In the 5:1 group the Piranha condition (-0.50 pt/s) typically ends before 15
# choices accrue (participants die early), so it is exempt there.
MIN_CHOICES = 15
ALL_CONDITIONS = ["Chicken", "Crab", "Turtle", "Piranha"]
EXEMPT_CONDITIONS = {"05": {"Piranha"}, "10": set()}


def eligible_subjects(preds, group, min_choices=MIN_CHOICES):
    """Subjects in `group` that, in every non-exempt condition, have at least
    `min_choices` choices and do not show exclusive preference for one
    alternative (i.e., neither all-SS nor all-LL). The exempt conditions are
    screened on neither criterion."""
    g = preds[(preds["group"] == group) & (preds["scope"] == "all")]
    obs = g.drop_duplicates(subset=["subject", "condition", "trial"])
    counts = obs.groupby(["subject", "condition"]).size().unstack(fill_value=0)
    rate = (obs.groupby(["subject", "condition"])["observed_choice"]
            .mean().unstack())
    required = [c for c in ALL_CONDITIONS
                if c not in EXEMPT_CONDITIONS.get(group, set())
                and c in counts.columns]
    enough = (counts[required] >= min_choices).all(axis=1)
    varied = ((rate[required] > 0) & (rate[required] < 1)).all(axis=1)
    ok = enough & varied
    return sorted(counts.index[ok].tolist())


def select_participants(preds, n_per_group=N_PER_GROUP, seed=SEED):
    """Group-balanced random draw of eligible subject IDs, sorted within group."""
    rng = np.random.default_rng(seed)
    chosen = []
    for grp in ["05", "10"]:
        subs = np.array(eligible_subjects(preds, grp))
        k = min(n_per_group, len(subs))
        exempt = ", ".join(sorted(EXEMPT_CONDITIONS.get(grp, set()))) or "none"
        print(f"  group {grp}: {len(subs)} eligible "
              f"(>= {MIN_CHOICES} choices/condition, non-exclusive; "
              f"exempt: {exempt})")
        pick = rng.choice(subs, size=k, replace=False)
        chosen.extend(sorted(pick.tolist()))
    return chosen


def _tex_escape(s):
    return s.replace("_", r"\_")


def write_appendix_tex(subjects, family_best, fig_rel_paths, out_path):
    """Emit a standalone Appendix_B.tex including one figure per participant."""
    fam_lines = []
    for fam in FAMILY_MODELS:
        if fam in family_best:
            label = pretty_model(family_best[fam]).replace("\n", " ")
            fam_lines.append(
                f"  \\item \\textbf{{{_tex_escape(fam)}}}: "
                f"{_tex_escape(label)}")
    fam_block = "\n".join(fam_lines)

    fig_blocks = []
    for sub, rel in zip(subjects, fig_rel_paths):
        grp = "5:1" if "_05_" in sub else "10:1"
        fig_blocks.append(
            "\\begin{figure}[p]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=\\textwidth]{{{rel}}}\n"
            f"  \\caption{{Trial-by-trial model fit for participant "
            f"{_tex_escape(sub)} (Group {grp}). Each panel is one earning-budget "
            f"condition. Black x marks are the participant's raw choices "
            f"(SS${{}}=1$, LL${{}}=0$), plotted just outside the $[0,1]$ band. "
            f"Gray lines are the predicted $P(\\mathrm{{SS}})$ of "
            f"the best-fitting model in each family.}}\n"
            "\\end{figure}\n")
    fig_block = "\n".join(fig_blocks)

    tex = rf"""\documentclass[12pt]{{article}}

\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{newtxtext,newtxmath}}  % Times New Roman text + math (APA)
\usepackage{{setspace}}
\usepackage{{microtype}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{enumitem}}

\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{6pt}}

\title{{Appendix B: Individual-Participant Model Fits}}
\author{{}}
\date{{}}

\begin{{document}}

\maketitle
\vspace{{-2.5em}}

The figures below show trial-by-trial model fits for a random, group-balanced
subset of {len(subjects)} participants ({N_PER_GROUP} from each group), drawn
with a fixed seed for reproducibility from those with at least {MIN_CHOICES}
recorded choices and no exclusive preference for a single alternative in each
condition (the Piranha condition is exempt in the 5:1 group, where it typically
ends before {MIN_CHOICES} choices accrue). Whereas the
figures in the main text
aggregate across participants, every model in this paper was fit at the
participant level; these panels illustrate the per-participant fits that
underlie those group summaries. Each figure is a $2\times2$ grid, one panel per
earning-budget condition. Black x marks show the participant's raw choice on
each trial (smaller--sooner ${{}}=1$, larger--later ${{}}=0$), plotted just
outside the $[0,1]$ band so they do not overlap the model lines. The gray lines
are the predicted probability of choosing the smaller--sooner option,
$P(\mathrm{{SS}})$,
from the best-performing model in each of the six families (lowest median
corrected Akaike information criterion at the pooled fit scope):

\begin{{itemize}}[noitemsep,topsep=0pt]
{fam_block}
\end{{itemize}}

\clearpage

{fig_block}

\end{{document}}
"""
    out_path.write_text(tex)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-group", type=int, default=N_PER_GROUP)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    preds = pd.read_parquet(config.RESULTS_DIR / "predictions.parquet")
    fits = pd.read_parquet(config.RESULTS_DIR / "fits_by_subject.parquet")

    family_best = best_model_per_family(fits, scope="all")
    print("Best model per family (lowest median AICc, pooled scope):")
    for fam in FAMILY_MODELS:
        if fam in family_best:
            print(f"  {fam:22s} -> {family_best[fam]}")
    models = [family_best[fam] for fam in FAMILY_MODELS if fam in family_best]

    subjects = select_participants(preds, args.n_per_group, args.seed)
    print(f"\nSelected {len(subjects)} participants: {subjects}")

    out_dir = config.FIG_DIR / "appendix_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear figures from prior runs so the directory only ever holds the
    # current selection (the drawn set changes with seed / eligibility rules).
    for stale in out_dir.glob("fig_participant_*.png"):
        stale.unlink()

    rel_paths = []
    for sub in subjects:
        out_path = out_dir / f"fig_participant_{sub}.png"
        p = participant_figure(preds, sub, models, out_path, scope="all")
        if p is not None:
            print(f"  Wrote {p}")
            rel_paths.append(f"data/results/figures/appendix_b/{out_path.name}")

    tex_path = write_appendix_tex(
        subjects, family_best, rel_paths, config.ROOT / "Appendix_B.tex")
    print(f"\nWrote {tex_path}")


if __name__ == "__main__":
    main()
