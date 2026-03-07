# Experiment Logs and Analysis

This directory contains the recorded results of **60 paired experiments** used to compare two NBV planning strategies:

- **L1+IK-only**: Layer-1 viewpoint generation with IK-feasibility filtering only
- **L1+L2**: Layer-1 + Layer-2 Action-Mode clustering and execution-aware selection

The experiments were organized as paired trials under approximately matched scene conditions.
For each `trial_id`, the two methods were executed on the same experimental setup so that planning outcome and motion-related metrics could be compared in a paired manner.

## Directory Overview

```text
experiment/
├── L1+IK_only/
│   ├── 1/
│   │   ├── mask_data.txt
│   │   └── nbv_executor_IK_only_xxx.csv
│   ├── 2/
│   └── ...
├── L1+L2/
│   ├── 1/
│   │   ├── mask_data.txt
│   │   └── nbv_executor_action_mode_xxx.csv
│   ├── 2/
│   └── ...
├── merge_experiments.py
├── paired_trial_analysis.py
├── make_three_line_table.py
├── all_experiments_summary.csv
├── paired_summary.csv
├── summary_table.csv
├── summary_table.tex
├── fig_paired_deltas.pdf
├── fig_paired_outcome.pdf
└── README.md


