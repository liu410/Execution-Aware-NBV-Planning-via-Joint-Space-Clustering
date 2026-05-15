#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# =========================
# 0. Figure style: match boxplot script
# =========================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.size"] = 9
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# =========================
# 1. Read data
# =========================
csv_path = "all_experiments_summary.csv"
df = pd.read_csv(csv_path)

non_numeric_cols = ["group", "csv_file", "csv_delim", "trial_tag"]
for col in df.columns:
    if col not in non_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

IK_NAME = "L1+IK_only"
L2_NAME = "L1+L2"

# =========================
# 2. Build paired table
# =========================
wide = df.pivot_table(
    index="trial_id",
    columns="group",
    values=[
        "planning_ok",
        "selected_dq_exec",
        "planning_time_sec",
        "sel_best_mode_m_limit",
    ],
    aggfunc="first"
)

rows = []
for trial_id in wide.index:
    try:
        ik_ok = wide.loc[trial_id, ("planning_ok", IK_NAME)]
        l2_ok = wide.loc[trial_id, ("planning_ok", L2_NAME)]
    except KeyError:
        continue

    if ik_ok == 1 and l2_ok == 1:
        def get_val(metric, method):
            try:
                return wide.loc[trial_id, (metric, method)]
            except KeyError:
                return np.nan

        rows.append({
            "trial_id": int(trial_id),

            "ik_d": get_val("selected_dq_exec", IK_NAME),
            "l2_d": get_val("selected_dq_exec", L2_NAME),

            "ik_t": get_val("planning_time_sec", IK_NAME),
            "l2_t": get_val("planning_time_sec", L2_NAME),

            "ik_muL": get_val("sel_best_mode_m_limit", IK_NAME),
            "l2_muL": get_val("sel_best_mode_m_limit", L2_NAME),
        })

paired_df = pd.DataFrame(rows).sort_values("trial_id")

# =========================
# 3. Plot settings
# =========================
plot_specs = [
    {
        "ylabel": r"$d(q)$ [rad]",
        "ik_col": "ik_d",
        "l2_col": "l2_d",
        "ylim": None,
    },
    {
        "ylabel": r"$t$ [s]",
        "ik_col": "ik_t",
        "l2_col": "l2_t",
        "ylim": None,
    },
    {
        "ylabel": r"$\mu_{L}$",
        "ik_col": "ik_muL",
        "l2_col": "l2_muL",
        "ylim": None,
    },
]

# Colorblind-friendly, journal-safe pair:
trial_color = "#4C78A8"   # muted blue for paired samples
mean_color  = "#D55E00"   # vermillion/orange-red for mean trend

# =========================
# 4. Plot
# =========================
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), dpi=300)
x = [0, 1]

for ax, spec in zip(axes, plot_specs):
    sub = paired_df.dropna(subset=[spec["ik_col"], spec["l2_col"]]).copy()

    # paired sample lines
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

    # mean trend line
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

    # annotations
    n = len(sub)
    ax.text(
        0.99, 0.99,
        f"n = {n}",
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

    # axes
    ax.set_xticks(x)
    ax.set_xticklabels(["L1+IK", "L1+L2"], fontsize=8)
    ax.set_xlim(-0.15, 1.15)

    ax.set_ylabel(spec["ylabel"], fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

    ax.grid(False)

    if spec["ylim"] is not None:
        ax.set_ylim(spec["ylim"])

plt.tight_layout()
fig.savefig("fig_paired_slope_3panel.pdf", bbox_inches="tight")
fig.savefig("fig_paired_slope_3panel.png", dpi=300, bbox_inches="tight")
fig.savefig("fig_paired_slope_3panel.svg", bbox_inches="tight")
plt.show()