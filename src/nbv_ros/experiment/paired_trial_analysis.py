#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
paired_trial_analysis.py

Paired analysis by trial_id between:
  - L1+IK-only
  - L1+L2

Core idea
---------
- Use trial_id to pair two methods under (approximately) matched scene conditions.
- Report a paired 2x2 contingency table of planning outcomes (ALL paired trials).
- For continuous metrics, compute deltas (L2 - IK) on the subset where BOTH methods planned successfully
  (recommended, --only_success 1).

Notation alignment (paper)
--------------------------
- Motion cost: d(q) = || W wrap(q - q0) ||_2  -> in CSV we reuse selected_dq_exec as d(q)
- Planning time: t_plan                      -> planning_time_sec
- Joint-limit margin: μ_L                    -> sel_best_mode_m_limit
- Singularity margin: μ_S                    -> sel_best_mode_m_sing (optional)

Paired deltas:
  Δd       = d_L2 - d_IK
  Δt_plan  = t_plan_L2 - t_plan_IK
  Δμ_L     = μ_L_L2 - μ_L_IK
  Δμ_S     = μ_S_L2 - μ_S_IK (optional)

Outputs
-------
1) Paired raw table:
   - paired_summary.csv

2) Figures:
   - fig_paired_deltas.pdf / .png
   - fig_paired_outcome.pdf / .png   (optional)

Usage
-----
cd D:\\experiment_2
python paired_trial_analysis.py --csv all_experiments_summary.csv --only_success 1
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.family"] = "Times New Roman"   # 或 "Times"
mpl.rcParams["mathtext.fontset"] = "stix"         # 数学符号更像 Times
mpl.rcParams["font.size"] = 9                     # 全局默认字号（再按需覆盖）
mpl.rcParams["axes.unicode_minus"] = False        # 防止负号乱码

METHOD_CANON = {
    "L1+IK_only": "L1+IK-only",
    "L1+IK only": "L1+IK-only",
    "ik_only": "L1+IK-only",
    "L1+L2": "L1+L2",
    "action_mode": "L1+L2",
}
METHODS = ["L1+IK-only", "L1+L2"]


def canonicalize_method(x: str) -> str:
    if not isinstance(x, str):
        return str(x)
    x = x.strip()
    return METHOD_CANON.get(x, x)


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def try_wilcoxon(x: np.ndarray):
    """Optional paired Wilcoxon signed-rank test. Returns p-value or None."""
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return None
    try:
        from scipy.stats import wilcoxon
        x2 = x[np.abs(x) > 1e-12]  # drop exact zeros
        if len(x2) < 5:
            return None
        _stat, p = wilcoxon(x2)
        return float(p)
    except Exception:
        return None


def parse_ylim(s: str):
    try:
        a, b = s.split(",")
        return (float(a), float(b))
    except Exception:
        raise ValueError(f"Bad ylim format: '{s}'. Expect 'low,high' like '-1.2,1.2'.")


def format_p(p):
    """Paper-friendly p formatting: 3 sig figs + unicode minus in exponent."""
    if p is None:
        return None
    s = f"{p:.3g}"
    return s.replace("e-", "e−")  # unicode minus


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", type=str, default="all_experiments_summary.csv")
    ap.add_argument("--out_csv", type=str, default="paired_summary.csv")

    ap.add_argument("--out_pdf", type=str, default="fig_paired_deltas.pdf")
    ap.add_argument("--out_png", type=str, default="fig_paired_deltas.png")

    ap.add_argument("--make_outcome_fig", type=int, default=1,
                    help="1: also plot paired outcome (success) for all paired trials; 0: skip.")
    ap.add_argument("--outcome_pdf", type=str, default="fig_paired_outcome.pdf")
    ap.add_argument("--outcome_png", type=str, default="fig_paired_outcome.png")

    ap.add_argument("--only_success", type=int, default=1,
                    help="1: keep only pairs where BOTH planning_ok=1 for delta plots; 0: keep all pairs.")
    ap.add_argument("--dedup_keep", type=str, default="last", choices=["first", "last"],
                    help="If multiple rows exist for same (trial_id, method), keep first or last.")

    # Strategy-B: manual y-axis limits (middle panel default tightened to "zoom in")
    ap.add_argument("--ylim_d", type=str, default="-6,2.5",
                    help="Manual y-limits for Δd plot, format 'low,high'. Example: -6,2.5")
    ap.add_argument("--ylim_t", type=str, default="-1.2,1.2",
                    help="Manual y-limits for Δt_plan plot, format 'low,high'. Example: -1.2,1.2")
    ap.add_argument("--ylim_muL", type=str, default="-0.8,1.0",
                    help="Manual y-limits for Δμ_L plot, format 'low,high'. Example: -0.8,1.0")

    args = ap.parse_args()

    YLIM_D = parse_ylim(args.ylim_d)
    YLIM_T = parse_ylim(args.ylim_t)
    YLIM_MUL = parse_ylim(args.ylim_muL)

    df = pd.read_csv(Path(args.csv))

    # method column
    if "group" in df.columns:
        df["method"] = df["group"].apply(canonicalize_method)
    elif "trial_tag" in df.columns:
        df["method"] = df["trial_tag"].apply(canonicalize_method)
    else:
        raise ValueError("Need 'group' or 'trial_tag' column for method.")

    if "trial_id" not in df.columns:
        raise ValueError("Missing trial_id column in CSV.")

    # Keep only methods of interest
    df = df[df["method"].isin(METHODS)].copy()

    # Columns for analysis
    keep_cols = [
        "trial_id", "method",
        "planning_ok",
        "selected_dq_exec",          # paper: d(q)
        "planning_time_sec",         # paper: t_plan
        "sel_best_mode_m_limit",     # paper: μ_L
        "sel_best_mode_m_sing",      # paper: μ_S (optional)
        "candidates_size",
        "ik_success_count",
        "ik_fail_count",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # numeric cleanup
    for c in keep_cols:
        if c not in ["trial_id", "method"] and c in df.columns:
            df[c] = safe_num(df[c])

    # split two methods then deduplicate by trial_id
    ik = df[df["method"] == "L1+IK-only"].copy()
    l2 = df[df["method"] == "L1+L2"].copy()

    ik = ik.sort_values(["trial_id"]).drop_duplicates("trial_id", keep=args.dedup_keep)
    l2 = l2.sort_values(["trial_id"]).drop_duplicates("trial_id", keep=args.dedup_keep)

    # inner join => paired by trial_id
    paired_all = pd.merge(
        l2, ik,
        on="trial_id",
        suffixes=("_l2", "_ik"),
        how="inner"
    )

    # ---- Paired outcome contingency (ALL paired trials) ----
    ok_l2 = safe_num(paired_all.get("planning_ok_l2", pd.Series(dtype=float))).fillna(0).astype(int)
    ok_ik = safe_num(paired_all.get("planning_ok_ik", pd.Series(dtype=float))).fillna(0).astype(int)

    both_ok = int(((ok_l2 == 1) & (ok_ik == 1)).sum())
    only_l2 = int(((ok_l2 == 1) & (ok_ik == 0)).sum())
    only_ik = int(((ok_l2 == 0) & (ok_ik == 1)).sum())
    both_fail = int(((ok_l2 == 0) & (ok_ik == 0)).sum())
    total_pairs = int(len(paired_all))

    # Per-method success rate over paired trials (same denominator)
    k_l2 = int((ok_l2 == 1).sum())
    k_ik = int((ok_ik == 1).sum())
    lo_l2, hi_l2 = wilson_ci(k_l2, total_pairs)
    lo_ik, hi_ik = wilson_ci(k_ik, total_pairs)

    print("\n=== Paired outcome contingency (by trial_id) ===")
    print(f"Total paired trial_id: {total_pairs}")
    print(f"  both planned ok     : {both_ok}")
    print(f"  only L2 planned ok  : {only_l2}")
    print(f"  only IK planned ok  : {only_ik}")
    print(f"  both failed         : {both_fail}")

    print("\n=== Success rate over paired trials (same denominator) ===")
    print(f"  L2: {k_l2}/{total_pairs} ({(k_l2/total_pairs*100 if total_pairs else 0):.1f}%) "
          f"CI95=[{lo_l2*100:.1f},{hi_l2*100:.1f}]%")
    print(f"  IK: {k_ik}/{total_pairs} ({(k_ik/total_pairs*100 if total_pairs else 0):.1f}%) "
          f"CI95=[{lo_ik*100:.1f},{hi_ik*100:.1f}]%")

    # ---- Outcome figure (ALL paired trials) ----
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
        ax2.text(0.98, 0.95, f"paired n = {total_pairs}", transform=ax2.transAxes,
                 ha="right", va="top", fontsize=9)

        plt.tight_layout()
        fig2.savefig(args.outcome_pdf, bbox_inches="tight")
        fig2.savefig(args.outcome_png, bbox_inches="tight")
        print(f"\nSaved: {args.outcome_pdf}, {args.outcome_png}")
        fig2.savefig("fig_paired_outcome.svg", bbox_inches="tight")

    # ---- Delta analysis table/plots ----
    paired = paired_all.copy()
    if args.only_success == 1:
        paired = paired[(ok_l2 == 1) & (ok_ik == 1)].copy()

    def delta(col):
        a = f"{col}_l2"
        b = f"{col}_ik"
        if a in paired.columns and b in paired.columns:
            return paired[a] - paired[b]
        return np.nan

    # Keep old column names (compat)
    paired["d_dq_exec"] = delta("selected_dq_exec")
    paired["d_time"] = delta("planning_time_sec")
    paired["d_m_limit"] = delta("sel_best_mode_m_limit")
    paired["d_m_sing"] = delta("sel_best_mode_m_sing")

    # Paper-aligned column names (preferred)
    paired["delta_d"] = paired["d_dq_exec"]            # Δd
    paired["delta_t_plan"] = paired["d_time"]          # Δt_plan
    paired["delta_mu_L"] = paired["d_m_limit"]         # Δμ_L
    paired["delta_mu_S"] = paired["d_m_sing"]          # Δμ_S (optional)

    paired.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    n_pairs_used = len(paired)
    print(f"\nPaired trials used in delta plots: {n_pairs_used} (only_success={args.only_success})")
    if n_pairs_used == 0:
        print("No paired data available with the current filter.")
        print(f"Saved: {args.out_csv}")
        return

    # Wilcoxon (optional) — compute using paper-aligned names
    p_d = try_wilcoxon(safe_num(paired["delta_d"]).values)
    p_t = try_wilcoxon(safe_num(paired["delta_t_plan"]).values)
    p_muL = try_wilcoxon(safe_num(paired["delta_mu_L"]).values)

    if p_d is not None or p_t is not None or p_muL is not None:
        print("\n=== Paired Wilcoxon p-values (L2 - IK) ===")
        if p_d is not None:
            print(f"  Δd         p = {p_d:.4g}")
        if p_t is not None:
            print(f"  Δt_plan    p = {p_t:.4g}")
        if p_muL is not None:
            print(f"  Δμ_L       p = {p_muL:.4g}")

    # Paper-style 1x3 plot (independent y-scales)
    # whis uses percentile range to avoid ultra-long whiskers
    metrics = [
        ("delta_d", r"$\Delta d$ [rad]", p_d, YLIM_D, (10, 90)),
        ("delta_t_plan", r"$\Delta t$ [s]", p_t, YLIM_T, (15, 85)),
        ("delta_mu_L", r"$\Delta \mu_{L}$", p_muL, YLIM_MUL, (10, 90)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), dpi=300)

    for ax, (key, ylab, pv, ylim, whis_range) in zip(axes, metrics):
        v = safe_num(paired[key]).dropna().values

        bp = ax.boxplot(
            [v],
            tick_labels=[""],
            showmeans=True,
            meanprops=dict(marker="^", markersize=4, zorder=6),
            boxprops=dict(linewidth=1.0, zorder=2),
            whiskerprops=dict(linewidth=1.0, zorder=2),
            capprops=dict(linewidth=1.0, zorder=2),
            medianprops=dict(linewidth=1.2, zorder=3),
            flierprops=dict(markersize=5, zorder=1),
            whis=whis_range,
        )

        # ✅ 1) 去掉 x 轴底部短竖线
        ax.set_xticks([])
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

        # ✅ 2) y 轴 padding（保留，但不再手动指定奇怪刻度）
        lo, hi = ylim
        pad = 0.04 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

        # ✅ 3) 恢复“标准刻度”（让 Matplotlib 自动选 1、0.5、0.2 这种漂亮刻度）
        #     关键：不要再 set_yticks(np.linspace(...))
        #     如果你还想更“整”，可以加这一句增强可读性（可选）：
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

        # ✅ 4) 让圈圈在箱子后面（不挡）
        for fl in bp["fliers"]:
            fl.set_zorder(0)

        ax.axhline(0, linewidth=1.0)
        ax.grid(False)
        ax.set_ylabel(ylab, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

        ax.text(0.98, 0.95, f"n = {len(v)}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8)
        if pv is not None:
            ax.text(0.02, 0.95, f"p = {format_p(pv)}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8)

    fig.suptitle("Paired Differences (L2 − IK)", fontsize=9, y=0.9)

    plt.tight_layout()
    fig.savefig(args.out_pdf, bbox_inches="tight")
    fig.savefig(args.out_png, bbox_inches="tight")
    fig.savefig("fig_paired_deltas.svg", bbox_inches="tight")
    print(f"\nSaved: {args.out_pdf}, {args.out_png}")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()