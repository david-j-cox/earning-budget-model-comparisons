"""Programmed task parameters and pipeline configuration.

Per the manuscript (Table 2). Reported values from Exp1 only — Exp2 is out of
scope for the initial ML companion paper.
"""

from pathlib import Path

# ---- Paths ----
# data/
#   raw/          source Exp1 spreadsheets
#   transformed/  long-format + engineered features (exp1_long, exp1_features)
#   results/      model outputs (parquet/CSV) and results/figures/
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRANSFORMED_DIR = DATA_DIR / "transformed"
RESULTS_DIR = DATA_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"

# Raw Exp1 spreadsheets (committed under data/raw/)
EXP1_FILES = {
    "05": RAW_DIR / "Exp1_05_to_1.xlsx",
    "10": RAW_DIR / "Exp1_10_to_1.xlsx",
}

# ---- Table 2 programmed task parameters (Exp1) ----
# Each condition: loss_rate (pt/s), R (required pts to survive), net_rate (pt/s if SS-only)
CONDITIONS = {
    "Chicken": {"loss_rate": 1 / 3.0,  "R": 100, "net_rate": +0.67, "sign": "positive"},
    "Crab":    {"loss_rate": 1 / 2.0,  "R": 150, "net_rate": +0.50, "sign": "positive"},
    "Turtle":  {"loss_rate": 1 / 0.95, "R": 315, "net_rate": -0.05, "sign": "negative"},
    "Piranha": {"loss_rate": 1 / 0.5,  "R": 600, "net_rate": -0.50, "sign": "negative"},
}

# Total time available per condition (T in Pietras notation)
T_SECONDS = 300

# LL:SS ratio per group (raw point amounts were scaled differently across groups;
# only the ratio is preserved and reported)
RATIO_BY_GROUP = {"05": 5, "10": 10}

# ---- Per-sheet row layouts ----
# The two Exp1 files have slightly different layouts (5-to-1 packs conditions
# tighter, 10-to-1 has blank rows). These were reverse-engineered from
# 01_aggregate_data.ipynb and verified on real participants.
SHEET_LAYOUT = {
    "05": {
        "start_rows": [0, 6, 12, 18],   # row of "Choice" for cond 1..4
        "label_rows": [0, 5, 11, 17],   # row containing the cond name in col 0
    },
    "10": {
        "start_rows": [0, 7, 14, 21],
        "label_rows": [0, 6, 13, 20],
    },
}

# Column where trial data begins (after Prop.Imm/value/Choice label columns)
DATA_COL_START = 3

# Per-block row offsets relative to start_row:
#   start_row+0: Choice
#   start_row+1: Bank
#   start_row+2: Delay - Lg.
#   start_row+3: Time Left
#   start_row+4: Prop. Last 10 (unused)
ROW_OFFSET_CHOICE = 0
ROW_OFFSET_BANK = 1
ROW_OFFSET_DELAY = 2
ROW_OFFSET_TIMELEFT = 3

# ---- Fitting parameters ----
N_STARTS = 10           # multi-start for non-convex MLE fits
N_JOBS = 6              # parallel workers
ROLLING_WINDOW = 10     # for local reward rates (kinetic)

# ---- Model output schema ----
RESULTS_SCHEMA = [
    "subject", "group", "condition", "scope",  # scope: "all" or condition name
    "model_family", "model", "n_params", "n_obs",
    "nll", "aic", "bic", "accuracy",
]
