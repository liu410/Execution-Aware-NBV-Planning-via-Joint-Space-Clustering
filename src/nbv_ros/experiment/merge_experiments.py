#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import csv
import glob

# Directory containing this script
root_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(root_dir, "all_experiments_summary.csv")

GROUPS = ["L1+IK_only", "L1+L2"]


def read_mask_txt(mask_file: str):
    """
    Read mask_data.txt.

    Example format (space-separated):
      84 5978.3 1.00 1.000 0.560 0.844 0.321 0.071 0.494

    Interpreted as:
      A, n, r, c, s, Z_obj, occ_infront, occ_ring, optional_extra
    """
    with open(mask_file, "r", encoding="utf-8") as f:
        vals = f.read().strip().split()

    if len(vals) < 8:
        raise ValueError(f"mask_data.txt values too few: {len(vals)} in {mask_file}")

    # Pad to at least 9 elements for safe indexing
    while len(vals) < 9:
        vals.append("")

    A, n, r, c, s, Z_obj, occ_infront, occ_ring, extra = vals[:9]
    return {
        "A": A,
        "n": n,
        "r": r,
        "c": c,
        "s": s,
        "Z_obj": Z_obj,
        "occ_infront": occ_infront,
        "occ_ring": occ_ring,
        "mask_extra": extra,
    }


def _detect_delimiter(sample: str):
    """
    Detect the delimiter from a text sample.

    Supported delimiters:
      - tab
      - comma
      - semicolon
    """
    tab_cnt = sample.count("\t")
    comma_cnt = sample.count(",")
    semi_cnt = sample.count(";")

    # Prefer tab if it is dominant
    if tab_cnt > 0 and tab_cnt >= comma_cnt and tab_cnt >= semi_cnt:
        return "\t"

    # Otherwise prefer comma if it is dominant
    if comma_cnt > 0 and comma_cnt >= tab_cnt and comma_cnt >= semi_cnt:
        return ","

    # Otherwise use semicolon if present
    if semi_cnt > 0:
        return ";"

    # Fallback
    return "\t"


def read_table_auto(csv_path: str):
    """
    Read a delimited table file that may actually be TSV, CSV, or semicolon-separated.

    Strategy:
      1. Read a sample and guess the delimiter.
      2. Parse the header.
      3. If the header looks suspicious (for example, only one column while
         containing other delimiters), retry with alternative delimiters.

    Returns:
      header: list[str] or None
      data_rows: list[list[str]]
      delimiter_used: str
    """
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)

        delim = _detect_delimiter(sample)
        reader = csv.reader(f, delimiter=delim)
        header = next(reader, None)

        if not header:
            return None, [], delim

        # If the header collapses into one column, retry other possible delimiters
        if len(header) == 1:
            h0 = header[0]
            candidates = []

            if "\t" in h0 and delim != "\t":
                candidates.append("\t")
            if "," in h0 and delim != ",":
                candidates.append(",")
            if ";" in h0 and delim != ";":
                candidates.append(";")

            for d2 in candidates:
                f.seek(0)
                reader2 = csv.reader(f, delimiter=d2)
                header2 = next(reader2, None)
                if header2 and len(header2) > 1:
                    header = header2
                    delim = d2
                    reader = reader2
                    break

        data_rows = []
        for row in reader:
            if not row:
                continue
            if not any(cell.strip() for cell in row):
                continue
            data_rows.append(row)

        return header, data_rows, delim


all_rows = []
all_fieldnames = set()


def safe_sort_key(x):
    """
    Sort folder names in a stable way.

    Numeric folder names are sorted by numeric value.
    Non-numeric folder names are placed after numeric ones and sorted lexicographically.
    """
    if x.isdigit():
        return (0, int(x))
    else:
        return (1, x)


for group in GROUPS:
    group_path = os.path.join(root_dir, group)
    if not os.path.isdir(group_path):
        print(f"[SKIP] missing group dir: {group_path}")
        continue

    trial_ids = sorted(
        [d for d in os.listdir(group_path)
         if os.path.isdir(os.path.join(group_path, d))],
        key=safe_sort_key
    )

    for trial_id in trial_ids:
        trial_path = os.path.join(group_path, trial_id)
        if not os.path.isdir(trial_path):
            continue

        mask_file = os.path.join(trial_path, "mask_data.txt")
        if not os.path.exists(mask_file):
            print(f"[SKIP] missing mask_data.txt: {trial_path}")
            continue

        try:
            mask_dict = read_mask_txt(mask_file)
        except Exception as e:
            print(f"[SKIP] bad mask_data.txt in {trial_path}: {e}")
            continue

        csv_files = sorted(glob.glob(os.path.join(trial_path, "*.csv")))
        if not csv_files:
            print(f"[SKIP] no csv in: {trial_path}")
            continue

        for csv_path in csv_files:
            header, data_rows, delim = read_table_auto(csv_path)
            csv_name = os.path.basename(csv_path)

            if not header:
                print(f"[SKIP] empty csv: {csv_path}")
                continue
            if not data_rows:
                print(f"[SKIP] csv has header but no data rows: {csv_path}")
                continue

            # Warn if the header still collapses into a single column
            if len(header) == 1:
                print(f"[WARN] header still 1-col after auto-detect: {csv_path}")
                print(f"       header[0][:120]={header[0][:120]}")

            for i, data in enumerate(data_rows):
                # Warn if a data row also collapses into one column and still contains delimiters
                if len(data) == 1 and (("\t" in data[0]) or ("," in data[0]) or (";" in data[0])):
                    print(f"[WARN] data row collapsed (possible delimiter or quoting issue): {csv_path} row={i}")
                    print(f"       data[0][:120]={data[0][:120]}")

                # Warn on column count mismatch instead of silently hiding the issue
                if len(data) != len(header):
                    print(f"[WARN] col mismatch: {csv_path} row={i} has {len(data)} cols, header has {len(header)} cols")

                # Pad or truncate the row so that it matches the header length
                if len(data) < len(header):
                    data = data + [""] * (len(header) - len(data))
                elif len(data) > len(header):
                    data = data[:len(header)]

                row = {
                    "group": group,
                    "trial_id": trial_id,
                    "csv_file": csv_name,
                    "csv_row_idx": str(i),
                    "csv_delim": {"\t": "TAB", ",": "COMMA", ";": "SEMI"}.get(delim, repr(delim)),
                }
                row.update(mask_dict)

                for k, v in zip(header, data):
                    key = k.strip()
                    row[key] = v.strip() if isinstance(v, str) else v

                all_rows.append(row)
                all_fieldnames.update(row.keys())

if not all_rows:
    print("No data was loaded. Please check the directory structure and file names.")
else:
    # Fixed front columns
    front = [
        "group", "trial_id", "csv_file", "csv_row_idx", "csv_delim",
        "A", "n", "r", "c", "s", "Z_obj", "occ_infront", "occ_ring", "mask_extra"
    ]
    rest = sorted([f for f in all_fieldnames if f not in front])
    fieldnames = [f for f in front if f in all_fieldnames] + rest

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Merge completed: {len(all_rows)} records")
    print(f"Output file: {output_file}")