"""Load Exp1 raw P-sheets into a long-format dataframe.

One row per (subject, condition, trial). Columns:
    subject       : str  e.g. "Exp1_05_P29"
    group         : str  "05" or "10"
    condition     : str  Chicken | Crab | Turtle | Piranha
    cond_order    : int  position of this condition in the participant's sequence (1..4)
    trial         : int  trial number within condition, starting at 1
    choice        : int  1 = SS (immediate), 0 = LL (delayed)
    bank          : float energy/points at moment of click
    delay_ll      : float current adjusting delay to LL (seconds)
    time_left     : float seconds remaining in condition at moment of click

Per-trial outcome (reward signal for RL) is derived in feature_engineering as
bank[t+1] - bank[t].
"""

import warnings

import numpy as np
import pandas as pd

from . import config

warnings.simplefilter("ignore")


def _read_p_sheet(xls, sheet_name, group):
    """Extract one participant's four-condition data from a P-sheet."""
    df = pd.read_excel(xls, sheet_name=sheet_name, engine="openpyxl")
    layout = config.SHEET_LAYOUT[group]
    rows = []

    for cond_order, (sr, lr) in enumerate(
        zip(layout["start_rows"], layout["label_rows"]), start=1
    ):
        # Need enough rows to read the block
        if sr + config.ROW_OFFSET_TIMELEFT >= len(df):
            continue

        # Condition name: for cond_order==1, it's the first column header;
        # otherwise it's in col 0 of the label_row.
        if cond_order == 1:
            cond_name = df.columns[0]
        else:
            cond_name = df.iloc[lr, 0]
        if not isinstance(cond_name, str):
            continue
        cond_name = cond_name.strip()
        if cond_name not in config.CONDITIONS:
            continue

        try:
            choice = df.iloc[sr + config.ROW_OFFSET_CHOICE,    config.DATA_COL_START:].dropna().values.astype(float)
            bank   = df.iloc[sr + config.ROW_OFFSET_BANK,      config.DATA_COL_START:].dropna().values.astype(float)
            delay  = df.iloc[sr + config.ROW_OFFSET_DELAY,     config.DATA_COL_START:].dropna().values.astype(float)
            tleft  = df.iloc[sr + config.ROW_OFFSET_TIMELEFT,  config.DATA_COL_START:].dropna().values.astype(float)
        except Exception:
            continue

        n = min(len(choice), len(bank), len(delay), len(tleft))
        if n == 0:
            continue

        for t in range(n):
            rows.append({
                "cond_order": cond_order,
                "condition": cond_name,
                "trial": t + 1,
                "choice": int(choice[t]),
                "bank": float(bank[t]),
                "delay_ll": float(delay[t]),
                "time_left": float(tleft[t]),
            })

    return rows


def load_exp1(verbose=True):
    """Load both 5-to-1 and 10-to-1 groups into a single long-format dataframe."""
    all_rows = []

    for group, fpath in config.EXP1_FILES.items():
        if not fpath.exists():
            raise FileNotFoundError(f"Missing source file: {fpath}")

        xls = pd.ExcelFile(fpath, engine="openpyxl")
        p_sheets = [s for s in xls.sheet_names if s.startswith("P (")]

        if verbose:
            print(f"  Group {group}: {len(p_sheets)} participant sheets in {fpath.name}")

        for sheet in p_sheets:
            pnum = sheet.split("(")[1].split(")")[0].strip()
            subject = f"Exp1_{group}_P{pnum}"
            for row in _read_p_sheet(xls, sheet, group):
                row["subject"] = subject
                row["group"] = group
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    # Drop subjects with zero trials in all conditions (corrupted entries)
    nonempty = df.groupby("subject")["trial"].size()
    keep = nonempty[nonempty > 0].index
    df = df[df["subject"].isin(keep)].reset_index(drop=True)

    # Order columns
    cols = ["subject", "group", "condition", "cond_order", "trial",
            "choice", "bank", "delay_ll", "time_left"]
    df = df[cols]

    if verbose:
        print(f"\nLoaded {len(df)} trial rows from {df['subject'].nunique()} subjects.")
        print(df.groupby(["group", "condition"]).size().unstack(fill_value=0))

    return df


def save_long(df, path=None):
    """Write the long-format frame to parquet."""
    path = path or (config.TRANSFORMED_DIR / "exp1_long.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    df = load_exp1(verbose=True)
    out = save_long(df)
    print(f"\nWrote {out}")
