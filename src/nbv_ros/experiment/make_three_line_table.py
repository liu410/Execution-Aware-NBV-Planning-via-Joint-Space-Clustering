#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_three_line_table.py

Read merged experiment CSV and produce a LaTeX three-line table (booktabs).

Metrics:
- Success (%) : success = (planning_ok==1 AND exec_ok==1)
- Δq (rad)    : selected_dq_exec (default computed on success-only trials)
- Time (s)    : planning_time_sec (default computed on success-only trials)
- m_limit     : sel_best_mode_m_limit (default computed on success-only trials)

Usage:
  python make_three_line_table.py --csv all_experiments_summary.csv

Outputs:
  - summary_table.tex   (LaTeX booktabs table)
  - summary_table.csv   (numeric table for sanity check)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

METHOD_CANON = {
    "L1+IK_only": "IK-only",
    "L1+IK only": "IK-only",
    "ik_only": "IK-only",
    "L1+IK-only": "IK-only",
    "IK-only": "IK-only",
    "L1+L2": "L2",
    "action_mode": "L2",
    "L2": "L2",
}

ORDER = ["IK-only", "L2"]

def canonicalize_method(x: str) -> str:
    if not isinstance(x, str):
        return str(x)
    x = x.strip()
    return METHOD_CANON.get(x, x)

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def fmt_pm(mean, std, decimals=2):
    if np.isnan(mean):
        return "--"
    if np.isnan(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="all_experiments_summary.csv")
    ap.add_argument("--out_tex", type=str, default="summary_table.tex")
    ap.add_argument("--out_csv", type=str, default="summary_table.csv")

    ap.add_argument("--success_on", type=str, default="planning_and_exec",
                    choices=["planning_and_exec", "planning_only"],
                    help="Success definition: planning_and_exec (planning_ok & exec_ok) or planning_only (planning_ok only).")

    ap.add_argument("--cont_on_success_only", type=int, default=1,
                    help="1: continuous metrics (dq/time/m_limit) computed on success-only trials; 0: compute on all trials (not recommended).")

    ap.add_argument("--decimals", type=int, default=2)
    args = ap.parse_args()

    df = pd.read_csv(Path(args.csv))

    # method column
    if "group" in df.columns:
        df["method"] = df["group"].apply(canonicalize_method)
    elif "trial_tag" in df.columns:
        df["method"] = df["trial_tag"].apply(canonicalize_method)
    else:
        raise ValueError("Need 'group' or 'trial_tag' column for method canonicalization.")

    # keep only relevant methods
    df = df[df["method"].isin(ORDER)].copy()

    # numeric cleanup
    for c in ["planning_ok", "exec_ok", "selected_dq_exec", "planning_time_sec", "sel_best_mode_m_limit"]:
        if c in df.columns:
            df[c] = safe_num(df[c])

    # success flag
    if args.success_on == "planning_only":
        df["success"] = (df["planning_ok"] == 1)
    else:
        df["success"] = (df["planning_ok"] == 1) & (df.get("exec_ok", 0) == 1)

    rows = []
    for m in ORDER:
        g = df[df["method"] == m].copy()
        N_all = len(g)
        succ = g["success"].fillna(False)
        succ_rate = 100.0 * succ.mean() if N_all > 0 else np.nan

        # subset for continuous stats
        if args.cont_on_success_only == 1:
            gg = g[succ].copy()
        else:
            gg = g

        dq_mean = gg["selected_dq_exec"].mean()
        dq_std  = gg["selected_dq_exec"].std(ddof=1)

        t_mean  = gg["planning_time_sec"].mean()
        t_std   = gg["planning_time_sec"].std(ddof=1)

        ml_mean = gg["sel_best_mode_m_limit"].mean()
        ml_std  = gg["sel_best_mode_m_limit"].std(ddof=1)

        rows.append({
            "Method": m,
            "N_all": N_all,
            "N_cont": len(gg),
            "Success (%)": succ_rate,
            "Δq (rad) mean": dq_mean,
            "Δq (rad) std": dq_std,
            "Time (s) mean": t_mean,
            "Time (s) std": t_std,
            "m_limit mean": ml_mean,
            "m_limit std": ml_std,
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    # Pretty strings for the three-line table
    dec = args.decimals
    out_disp = pd.DataFrame({
        "Method": out["Method"],
        "Success (%)": out["Success (%)"].map(lambda x: "--" if np.isnan(x) else f"{x:.0f}%"),
        "Δq (rad)": [fmt_pm(a, b, dec) for a, b in zip(out["Δq (rad) mean"], out["Δq (rad) std"])],
        "Time (s)": [fmt_pm(a, b, dec) for a, b in zip(out["Time (s) mean"], out["Time (s) std"])],
        "m_limit": [fmt_pm(a, b, dec) for a, b in zip(out["m_limit mean"], out["m_limit std"])],
    })

    # Print a text preview (copy-friendly)
    print("\n=== Three-line table preview ===")
    print(out_disp.to_string(index=False))

    # Write LaTeX booktabs table
    # Note: requires \\usepackage{booktabs} in your LaTeX preamble.
    caption = "Overall statistics (mean ± std)."
    label = "tab:overall_stats"

    tex_lines = []
    tex_lines.append(r"\begin{table}[t]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\caption{" + caption + r"}")
    tex_lines.append(r"\label{" + label + r"}")
    tex_lines.append(r"\begin{tabular}{lcccc}")
    tex_lines.append(r"\toprule")
    tex_lines.append(r"Method & Success (\%) & $\Delta q$ (rad) & Time (s) & $m_{\mathrm{limit}}$ \\")
    tex_lines.append(r"\midrule")

    for _, r in out_disp.iterrows():
        tex_lines.append(
            f"{r['Method']} & {r['Success (%)']} & {r['Δq (rad)']} & {r['Time (s)']} & {r['m_limit']} \\\\"
        )

    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    # optional footnote about denominators
    if args.cont_on_success_only == 1:
        tex_lines.append(r"\vspace{0.5mm}")
        tex_lines.append(r"{\footnotesize Continuous metrics are computed on successful trials only.}")
    tex_lines.append(r"\end{table}")

    Path(args.out_tex).write_text("\n".join(tex_lines), encoding="utf-8")

    print(f"\nSaved: {args.out_tex}")
    print(f"Saved: {args.out_csv}")

if __name__ == "__main__":
    main()