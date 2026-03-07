#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
paired_trial_analysis.py

Paired analysis by trial_id between:
  - L1+IK-only
  - L1+L2

Core idea
---------
- Use trial_id to pair two methods under approximately matched scene conditions.
- Report a paired 2×2 contingency table of planning outcomes for all paired trials.
- For continuous metrics, compute deltas (L2 - IK) on the subset where both
  methods successfully planned (recommended: --only_success 1).

Notation alignment (paper)
--------------------------
- Motion cost: d(q) = || W wrap(q - q0) ||_2
  In the CSV file this corresponds to: selected_dq_exec
- Planning time: t_plan
  In the CSV file this corresponds to: planning_time_sec
- Joint-limit margin: μ_L
  In the CSV file this corresponds to: sel_best_mode_m_limit
- Singularity margin: μ_S (optional)
  In the CSV file this corresponds to: sel_best_mode_m_sing

Paired deltas
-------------
Δd        = d_L2 - d_IK
Δt_plan   = t_plan_L2 - t_plan_IK
Δμ_L      = μ_L_L2 - μ_L_IK
Δμ_S      = μ_S_L2 - μ_S_IK (optional)

Outputs
-------
1) Paired raw table
   - paired_summary.csv

2) Figures
   - fig_paired_deltas.pdf / .png
   - fig_paired_outcome.pdf / .png (optional)

Usage
-----
cd ~/catkin_ws/src/nbv_ros/experiment
python3 paired_trial_analysis.py --csv all_experiments_summary.csv --only_success 1
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["DejaVu Serif", "Liberation Serif", "Nimbus Roman"]
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.size"] = 9
mpl.rcParams["axes.unicode_minus"] = False

METHOD_CANON = {
    "L1+IK_only": "L1+IK-only",
    "L1+IK only": "L1+IK-only",
    "ik_only": "L1+IK-only",
    "L1+L2": "L1+L2",
    "action_mode": "L1+L2",
}
METHODS = ["L1+IK-only", "L1+L2"]


def canonicalize_method(x: str) -> str:
    """Normalize different method name variants to canonical names."""
    if not isinstance(x, str):
        return str(x)
    x = x.strip()
    return METHOD_CANON.get(x, x)


def safe_num(s: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric values with NaN for invalid entries."""
    return pd.to_numeric(s, errors="coerce")


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Compute Wilson confidence interval for binomial proportion."""
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def try_wilcoxon(x: np.ndarray):
    """
    Optional paired Wilcoxon signed-rank test.
    Returns the p-value if the test is valid, otherwise None.
    """
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return None
    try:
        from scipy.stats import wilcoxon
        x2 = x[np.abs(x) > 1e-12]  # remove exact zero differences
        if len(x2) < 5:
            return None
        _stat, p = wilcoxon(x2)
        return float(p)
    except Exception:
        return None


def parse_ylim(s: str):
    """Parse manual y-axis limits provided as 'low,high'."""
    try:
        a, b = s.split(",")
        return (float(a), float(b))
    except Exception:
        raise ValueError(f"Bad ylim format: '{s}'. Expect 'low,high' like '-1.2,1.2'.")


def format_p(p):
    """Format p-values for paper-style output (3 significant digits)."""
    if p is None:
        return None
    s = f"{p:.3g}"
    return s.replace("e-", "e−")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", type=str, default="all_experiments_summary.csv")
    ap.add_argument("--out_csv", type=str, default="paired_summary.csv")

    ap.add_argument("--out_pdf", type=str, default="fig_paired_deltas.pdf")
    ap.add_argument("--out_png", type=str, default="fig_paired_deltas.png")

    ap.add_argument(
        "--make_outcome_fig",
        type=int,
        default=1,
        help="1: also generate the paired outcome (success) figure; 0: skip."
    )

    ap.add_argument("--outcome_pdf", type=str, default="fig_paired_outcome.pdf")
    ap.add_argument("--outcome_png", type=str, default="fig_paired_outcome.png")

    ap.add_argument(
        "--only_success",
        type=int,
        default=1,
        help="1: use only pairs where both planning_ok=1 for delta plots."
    )

    ap.add_argument(
        "--dedup_keep",
        type=str,
        default="last",
        choices=["first", "last"],
        help="If multiple rows exist for the same (trial_id, method), keep first or last."
    )

    # Manual y-axis limits for the three delta plots
    ap.add_argument("--ylim_d", type=str, default="-6,2.5")
    ap.add_argument("--ylim_t", type=str, default="-1.2,1.2")
    ap.add_argument("--ylim_muL", type=str, default="-0.8,1.0")

    args = ap.parse_args()

    YLIM_D = parse_ylim(args.ylim_d)
    YLIM_T = parse_ylim(args.ylim_t)
    YLIM_MUL = parse_ylim(args.ylim_muL)

    df = pd.read_csv(Path(args.csv))

    # Determine which column indicates the method
    if "group" in df.columns:
        df["method"] = df["group"].apply(canonicalize_method)
    elif "trial_tag" in df.columns:
        df["method"] = df["trial_tag"].apply(canonicalize_method)
    else:
        raise ValueError("CSV must contain 'group' or 'trial_tag' column.")

    if "trial_id" not in df.columns:
        raise ValueError("Missing trial_id column in CSV.")

    # Keep only the two methods of interest
    df = df[df["method"].isin(METHODS)].copy()

    # Columns used for analysis
    keep_cols = [
        "trial_id", "method",
        "planning_ok",
        "selected_dq_exec",
        "planning_time_sec",
        "sel_best_mode_m_limit",
        "sel_best_mode_m_sing",
        "candidates_size",
        "ik_success_count",
        "ik_fail_count",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Convert numeric columns safely
    for c in keep_cols:
        if c not in ["trial_id", "method"] and c in df.columns:
            df[c] = safe_num(df[c])

    # Split the two methods
    ik = df[df["method"] == "L1+IK-only"].copy()
    l2 = df[df["method"] == "L1+L2"].copy()

    # Deduplicate by trial_id
    ik = ik.sort_values(["trial_id"]).drop_duplicates("trial_id", keep=args.dedup_keep)
    l2 = l2.sort_values(["trial_id"]).drop_duplicates("trial_id", keep=args.dedup_keep)

    # Pair the trials using inner join
    paired_all = pd.merge(
        l2, ik,
        on="trial_id",
        suffixes=("_l2", "_ik"),
        how="inner"
    )

    # Planning success columns
    ok_l2 = safe_num(paired_all.get("planning_ok_l2", pd.Series(dtype=float))).fillna(0).astype(int)
    ok_ik = safe_num(paired_all.get("planning_ok_ik", pd.Series(dtype=float))).fillna(0).astype(int)

    both_ok = int(((ok_l2 == 1) & (ok_ik == 1)).sum())
    only_l2 = int(((ok_l2 == 1) & (ok_ik == 0)).sum())
    only_ik = int(((ok_l2 == 0) & (ok_ik == 1)).sum())
    both_fail = int(((ok_l2 == 0) & (ok_ik == 0)).sum())
    total_pairs = int(len(paired_all))

    # Success rates with Wilson confidence intervals
    k_l2 = int((ok_l2 == 1).sum())
    k_ik = int((ok_ik == 1).sum())

    lo_l2, hi_l2 = wilson_ci(k_l2, total_pairs)
    lo_ik, hi_ik = wilson_ci(k_ik, total_pairs)

    print("\nPaired outcome contingency (by trial_id)")
    print(f"Total paired trials: {total_pairs}")
    print(f"Both success : {both_ok}")
    print(f"L2 only      : {only_l2}")
    print(f"IK only      : {only_ik}")
    print(f"Both fail    : {both_fail}")

    print("\nSuccess rate over paired trials")
    print(f"L2: {k_l2}/{total_pairs} ({(k_l2/total_pairs*100 if total_pairs else 0):.1f}%) "
          f"CI95=[{lo_l2*100:.1f},{hi_l2*100:.1f}]%")
    print(f"IK: {k_ik}/{total_pairs} ({(k_ik/total_pairs*100 if total_pairs else 0):.1f}%) "
          f"CI95=[{lo_ik*100:.1f},{hi_ik*100:.1f}]%")

    # Outcome figure (all paired trials)
    if args.make_outcome_fig == 1 and total_pairs > 0:

        outcome = np.zeros(total_pairs, dtype=int)
        outcome[(ok_l2.values == 1) & (ok_ik.values == 0)] = +1
        outcome[(ok_l2.values == 0) & (ok_ik.values == 1)] = -1

        cats = [-1, 0, +1]
        counts = [int((outcome == c).sum()) for c in cats]

        fig2 = plt.figure(figsize=(3.6, 2.2), dpi=300)
        ax2 = plt.gca()

        ax2.bar(range(3), counts, alpha=0.85, edgecolor="black", linewidth=0.8)

        ax2.set_xticks(range(3))
        ax2.set_xticklabels(["IK wins", "Tie", "L2 wins"], fontsize=8)

        ax2.tick_params(axis="y", labelsize=8)
        ax2.set_ylabel("Count", fontsize=9)

        ax2.text(0.98, 0.95, f"paired n = {total_pairs}",
                 transform=ax2.transAxes,
                 ha="right", va="top", fontsize=9)

        plt.tight_layout()

        fig2.savefig(args.outcome_pdf, bbox_inches="tight")
        fig2.savefig(args.outcome_png, bbox_inches="tight")
        fig2.savefig("fig_paired_outcome.svg", bbox_inches="tight")

        print(f"\nSaved: {args.outcome_pdf}, {args.outcome_png}")

    # Delta analysis
    paired = paired_all.copy()

    if args.only_success == 1:
        paired = paired[(ok_l2 == 1) & (ok_ik == 1)].copy()

    def delta(col):
        """Compute paired difference L2 - IK."""
        a = f"{col}_l2"
        b = f"{col}_ik"
        if a in paired.columns and b in paired.columns:
            return paired[a] - paired[b]
        return np.nan

    paired["delta_d"] = delta("selected_dq_exec")
    paired["delta_t_plan"] = delta("planning_time_sec")
    paired["delta_mu_L"] = delta("sel_best_mode_m_limit")
    paired["delta_mu_S"] = delta("sel_best_mode_m_sing")

    paired.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()