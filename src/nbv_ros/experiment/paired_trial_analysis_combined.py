#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
paired_trial_analysis_combined.py

Combined paired analysis figure:
Top row:
  - Boxplots of paired differences: Δd, Δt, Δμ_L

Bottom row:
  - Paired slope plots of raw paired values: d(q), t, μ_L

Pairing:
  - By trial_id
  - Continuous metrics use pairs where both methods planning_ok = 1
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# =========================
# 0. Figure style
# =========================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.size"] = 9
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


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
        x2 = x[np.abs(x) > 1e-12]
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
        raise ValueError(
            f"Bad ylim format: '{s}'. Expect 'low,high', e.g. '-6,2.5'."
        )


def format_p(p):
    if p is None:
        return None
    s = f"{p:.3g}"
    return s.replace("e-", "e−")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", type=str, default="all_experiments_summary.csv")
    ap.add_argument("--out_csv", type=str, default="paired_summary.csv")

    ap.add_argument("--out_pdf", type=str, default="fig_paired_combined.pdf")
    ap.add_argument("--out_png", type=str, default="fig_paired_combined.png")
    ap.add_argument("--out_svg", type=str, default="fig_paired_combined.svg")

    ap.add_argument("--only_success", type=int, default=1)
    ap.add_argument("--dedup_keep", type=str, default="last", choices=["first", "last"])

    # Top-row boxplot limits
    ap.add_argument("--ylim_d", type=str, default="-6,2.5")
    ap.add_argument("--ylim_t", type=str, default="-1.2,1.2")
    ap.add_argument("--ylim_muL", type=str, default="-0.8,1.0")

    # Bottom-row slope plot limits. Use empty string "" for auto.
    ap.add_argument("--slope_ylim_d", type=str, default="")
    ap.add_argument("--slope_ylim_t", type=str, default="")
    ap.add_argument("--slope_ylim_muL", type=str, default="")

    args = ap.parse_args()

    YLIM_D = parse_ylim(args.ylim_d)
    YLIM_T = parse_ylim(args.ylim_t)
    YLIM_MUL = parse_ylim(args.ylim_muL)

    SLOPE_YLIM_D = parse_ylim(args.slope_ylim_d) if args.slope_ylim_d else None
    SLOPE_YLIM_T = parse_ylim(args.slope_ylim_t) if args.slope_ylim_t else None
    SLOPE_YLIM_MUL = parse_ylim(args.slope_ylim_muL) if args.slope_ylim_muL else None

    # =========================
    # 1. Load and clean data
    # =========================
    df = pd.read_csv(Path(args.csv))

    if "group" in df.columns:
        df["method"] = df["group"].apply(canonicalize_method)
    elif "trial_tag" in df.columns:
        df["method"] = df["trial_tag"].apply(canonicalize_method)
    else:
        raise ValueError("Need 'group' or 'trial_tag' column for method.")

    if "trial_id" not in df.columns:
        raise ValueError("Missing trial_id column in CSV.")

    df = df[df["method"].isin(METHODS)].copy()

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

    for c in keep_cols:
        if c not in ["trial_id", "method"] and c in df.columns:
            df[c] = safe_num(df[c])

    ik = df[df["method"] == "L1+IK-only"].copy()
    l2 = df[df["method"] == "L1+L2"].copy()

    ik = ik.sort_values(["trial_id"]).drop_duplicates("trial_id", keep=args.dedup_keep)
    l2 = l2.sort_values(["trial_id"]).drop_duplicates("trial_id", keep=args.dedup_keep)

    paired_all = pd.merge(
        l2, ik,
        on="trial_id",
        suffixes=("_l2", "_ik"),
        how="inner"
    )

    ok_l2 = safe_num(paired_all["planning_ok_l2"]).fillna(0).astype(int)
    ok_ik = safe_num(paired_all["planning_ok_ik"]).fillna(0).astype(int)

    both_ok = int(((ok_l2 == 1) & (ok_ik == 1)).sum())
    only_l2 = int(((ok_l2 == 1) & (ok_ik == 0)).sum())
    only_ik = int(((ok_l2 == 0) & (ok_ik == 1)).sum())
    both_fail = int(((ok_l2 == 0) & (ok_ik == 0)).sum())
    total_pairs = int(len(paired_all))

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

    print("\n=== Success rate over paired trials ===")
    print(f"  L2: {k_l2}/{total_pairs} ({k_l2 / total_pairs * 100:.1f}%) "
          f"CI95=[{lo_l2 * 100:.1f},{hi_l2 * 100:.1f}]%")
    print(f"  IK: {k_ik}/{total_pairs} ({k_ik / total_pairs * 100:.1f}%) "
          f"CI95=[{lo_ik * 100:.1f},{hi_ik * 100:.1f}]%")

    # =========================
    # 2. Delta analysis data
    # =========================
    paired = paired_all.copy()
    if args.only_success == 1:
        paired = paired[(ok_l2 == 1) & (ok_ik == 1)].copy()

    def delta(col):
        return paired[f"{col}_l2"] - paired[f"{col}_ik"]

    paired["delta_d"] = delta("selected_dq_exec")
    paired["delta_t_plan"] = delta("planning_time_sec")
    paired["delta_mu_L"] = delta("sel_best_mode_m_limit")
    if "sel_best_mode_m_sing_l2" in paired.columns and "sel_best_mode_m_sing_ik" in paired.columns:
        paired["delta_mu_S"] = delta("sel_best_mode_m_sing")

    paired.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"\nPaired trials used in continuous metric plots: {len(paired)}")

    p_d = try_wilcoxon(safe_num(paired["delta_d"]).values)
    p_t = try_wilcoxon(safe_num(paired["delta_t_plan"]).values)
    p_muL = try_wilcoxon(safe_num(paired["delta_mu_L"]).values)

    print("\n=== Paired Wilcoxon p-values (L2 - IK) ===")
    if p_d is not None:
        print(f"  Δd      p = {p_d:.4g}")
    if p_t is not None:
        print(f"  Δt      p = {p_t:.4g}")
    if p_muL is not None:
        print(f"  Δμ_L    p = {p_muL:.4g}")

    # =========================
    # 3. Combined figure: 2 x 3
    # =========================
    trial_color = "#4C78A8"
    mean_color = "#D55E00"

    fig, axes = plt.subplots(
        2, 3,
        figsize=(7.0, 6),
        dpi=300,
        gridspec_kw={"height_ratios": [1.2, 2.4], "hspace": 0.20, "wspace": 0.36}
    )

    # ---------- Top row: boxplots ----------
    box_metrics = [
        ("delta_d", r"$\Delta d$ [rad]", p_d, YLIM_D, (10, 90)),
        ("delta_t_plan", r"$\Delta t$ [s]", p_t, YLIM_T, (15, 85)),
        ("delta_mu_L", r"$\Delta \mu_{L}$", p_muL, YLIM_MUL, (10, 90)),
    ]

    for ax, (key, ylab, pv, ylim, whis_range) in zip(axes[0], box_metrics):
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

        ax.set_xticks([])
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

        lo, hi = ylim
        pad = 0.04 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

        for fl in bp["fliers"]:
            fl.set_zorder(0)

        ax.axhline(0, linewidth=1.0)
        ax.grid(False)
        ax.set_ylabel(ylab, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

        ax.text(
            0.98, 0.98,
            f"n = {len(v)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8
        )

        if pv is not None:
            ax.text(
                0.02, 0.98,
                f"p = {format_p(pv)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8
            )

    # ---------- Bottom row: paired slope plots ----------
    slope_specs = [
        {
            "ylabel": r"$d(q)$ [rad]",
            "l2_col": "selected_dq_exec_l2",
            "ik_col": "selected_dq_exec_ik",
            "ylim": SLOPE_YLIM_D,
        },
        {
            "ylabel": r"$t$ [s]",
            "l2_col": "planning_time_sec_l2",
            "ik_col": "planning_time_sec_ik",
            "ylim": SLOPE_YLIM_T,
        },
        {
            "ylabel": r"$\mu_{L}$",
            "l2_col": "sel_best_mode_m_limit_l2",
            "ik_col": "sel_best_mode_m_limit_ik",
            "ylim": SLOPE_YLIM_MUL,
        },
    ]

    x = [0, 1]

    for ax, spec in zip(axes[1], slope_specs):
        sub = paired.dropna(subset=[spec["ik_col"], spec["l2_col"]]).copy()

        for _, row in sub.iterrows():
            ax.plot(
                x,
                [row[spec["ik_col"]], row[spec["l2_col"]]],
                color=trial_color,
                marker="o",
                markersize=0.4,
                linewidth=0.6,
                alpha=0.7,
                zorder=1
            )

        mean_ik = sub[spec["ik_col"]].mean()
        mean_l2 = sub[spec["l2_col"]].mean()
        mean_delta = mean_l2 - mean_ik

        ax.plot(
            x,
            [mean_ik, mean_l2],
            color=mean_color,
            marker="o",
            markersize=3.0,
            linewidth=1.5,
            alpha=0.9,
            zorder=5
        )

        ax.text(
            0.98, 0.99,
            f"n = {len(sub)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8
        )

        ax.text(
            0.03, 0.001,
            f"mean Δ = {mean_delta:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8
        )

        ax.set_xticks(x)
        ax.set_xticklabels(["L1+IK", "L1+L2"], fontsize=8)
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylabel(spec["ylabel"], fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(False)

        if spec["ylim"] is not None:
            ax.set_ylim(spec["ylim"])

    # =========================
    # 4. Row titles and layout
    # =========================
    fig.subplots_adjust(
        top=0.90,
        bottom=0.08,
        left=0.08,
        right=0.98,
        hspace=0.92,
        wspace=0.36
    )

    fig.text(
        0.5, 0.955,
        "Paired Differences (L2 − IK)",
        ha="center",
        va="top",
        fontsize=12
    )

    fig.text(
        0.5, 0.620,
        "Paired Slope Plots (IK vs. L2)",
        ha="center",
        va="center",
        fontsize=12
    )

    fig.savefig(args.out_pdf, bbox_inches="tight")
    fig.savefig(args.out_png, dpi=300, bbox_inches="tight")
    fig.savefig(args.out_svg, bbox_inches="tight")

    print(f"\nSaved: {args.out_pdf}")
    print(f"Saved: {args.out_png}")
    print(f"Saved: {args.out_svg}")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()