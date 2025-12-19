#!/usr/bin/env python3
"""
Aggregate group probabilities by lemma (auto-detect group columns).

Input: CSV produced by roberta_mlm_on_verbs.py with groups enabled.
Output (default): CSV with one row per lemma and mean group probability per group.

If --excel is provided, also write an .xlsx with two sheets:
  1) lemma_to_groups: lemma -> group percentages (Excel % formatting) + count
  2) groups_ranked: for each group, two columns (lemma, pct) sorted by decreasing pct

Defaults:
  - If output_csv is omitted, writes: <input_basename>.group_means.csv
  - If --excel is provided without a filename, writes: <input_basename>.group_means.xlsx

Auto-detection:
  - If --group-cols is provided, use it.
  - Otherwise infer group columns by taking all columns after the last prob_k column.

Requirements for --excel:
  pip install openpyxl
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


PROB_COL_RE = re.compile(r"^prob_(\d+)$")


def infer_group_cols(fieldnames: List[str]) -> List[str]:
    """Infer group columns as those appearing after the last prob_k column."""
    last_prob_idx = -1
    for i, name in enumerate(fieldnames):
        if PROB_COL_RE.match(name.strip()):
            last_prob_idx = i
    if last_prob_idx == -1:
        return []
    return [c for c in fieldnames[last_prob_idx + 1 :] if c.strip()]


def default_output_csv_name(input_csv: str) -> str:
    base, _ = os.path.splitext(input_csv)
    return f"{base}.group_means.csv"


def default_output_xlsx_name(input_csv: str) -> str:
    base, _ = os.path.splitext(input_csv)
    return f"{base}.group_means.xlsx"


def safe_sheet_title(s: str) -> str:
    # Excel sheet name limits: <=31 chars, no : \ / ? * [ ]
    bad = r'[]:*?/\\'
    out = "".join("_" if ch in bad else ch for ch in s)
    return out[:31] if len(out) > 31 else out


def write_excel(
    xlsx_path: str,
    lemma_col: str,
    group_cols: List[str],
    counts: Dict[str, int],
    means: Dict[str, Dict[str, float]],  # means[group][lemma] -> float
) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.utils import get_column_letter
        from openpyxl.styles import Font, Alignment
    except ImportError as e:
        raise SystemExit("openpyxl is required for --excel. Install with: pip install openpyxl") from e

    wb = Workbook()

    # ---------------- Sheet 1: lemma_to_groups ----------------
    ws1 = wb.active
    ws1.title = safe_sheet_title("lemma_to_groups")

    header1 = [lemma_col] + group_cols + ["count"]
    ws1.append(header1)

    bold = Font(bold=True)
    for cell in ws1[1]:
        cell.font = bold
        cell.alignment = Alignment(vertical="center")

    # Write rows
    for lemma in sorted(counts.keys()):
        row = [lemma]
        for g in group_cols:
            row.append(means[g].get(lemma, 0.0))
        row.append(counts[lemma])
        ws1.append(row)

    # Format % columns
    # lemma_col at A; groups are B..; count is last
    for j in range(2, 2 + len(group_cols)):
        for i in range(2, ws1.max_row + 1):
            ws1.cell(row=i, column=j).number_format = "0.00%"

    # Freeze panes and autofilter
    ws1.freeze_panes = "A2"
    ws1.auto_filter.ref = f"A1:{get_column_letter(ws1.max_column)}{ws1.max_row}"

    # Reasonable column widths
    ws1.column_dimensions["A"].width = max(12, min(40, max(len(lemma_col), 12)))
    for idx, g in enumerate(group_cols, start=2):
        ws1.column_dimensions[get_column_letter(idx)].width = max(12, min(24, len(g) + 2))
    ws1.column_dimensions[get_column_letter(ws1.max_column)].width = 10  # count

    # ---------------- Sheet 2: groups_ranked ----------------
    ws2 = wb.create_sheet(title=safe_sheet_title("groups_ranked"))

    # Build sorted lists per group: [(lemma, pct), ...] descending pct
    per_group_sorted: Dict[str, List[Tuple[str, float]]] = {}
    max_len = 0
    for g in group_cols:
        items = [(lemma, means[g].get(lemma, 0.0)) for lemma in counts.keys()]
        items.sort(key=lambda x: x[1], reverse=True)
        per_group_sorted[g] = items
        max_len = max(max_len, len(items))

    # Header: two cols per group
    header2: List[str] = []
    for g in group_cols:
        header2.extend([f"{g}_lemma", f"{g}_pct"])
    ws2.append(header2)
    for cell in ws2[1]:
        cell.font = bold
        cell.alignment = Alignment(vertical="center")

    # Rows: align by rank index
    for r in range(max_len):
        row: List[Optional[object]] = []
        for g in group_cols:
            lemma, pct = per_group_sorted[g][r]
            row.extend([lemma, pct])
        ws2.append(row)

    # Format every *_pct column as percentage
    # pct columns are 2,4,6,... (1-indexed in Excel columns)
    for col in range(2, 2 * len(group_cols) + 1, 2):
        for i in range(2, ws2.max_row + 1):
            ws2.cell(row=i, column=col).number_format = "0.00%"

    ws2.freeze_panes = "A2"
    ws2.auto_filter.ref = f"A1:{get_column_letter(ws2.max_column)}{ws2.max_row}"

    # Column widths for pairs
    for k, g in enumerate(group_cols):
        lemma_col_idx = 1 + 2 * k
        pct_col_idx = 2 + 2 * k
        ws2.column_dimensions[get_column_letter(lemma_col_idx)].width = max(12, min(28, len(g) + 6))
        ws2.column_dimensions[get_column_letter(pct_col_idx)].width = 12

    wb.save(xlsx_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", help="MLM output CSV with group columns appended")
    ap.add_argument(
        "output_csv",
        nargs="?",
        default=None,
        help="Output CSV (default: <input>.group_means.csv)",
    )
    ap.add_argument("--lemma-col", default="lemma", help="Lemma column name (default: lemma)")
    ap.add_argument(
        "--group-cols",
        nargs="+",
        default=None,
        help="Optional explicit group column names; otherwise inferred from header",
    )
    ap.add_argument(
        "--excel",
        nargs="?",
        const="__AUTO__",
        default=None,
        help="Also write an Excel .xlsx (optional filename; default: <input>.group_means.xlsx)",
    )
    ap.add_argument("--encoding", default="utf-8")
    args = ap.parse_args()

    output_csv = args.output_csv or default_output_csv_name(args.input_csv)
    excel_path: Optional[str] = None
    if args.excel is not None:
        excel_path = default_output_xlsx_name(args.input_csv) if args.excel == "__AUTO__" else args.excel
        if not excel_path.lower().endswith(".xlsx"):
            excel_path += ".xlsx"

    # Pass 1: read + accumulate sums and counts
    with open(args.input_csv, newline="", encoding=args.encoding) as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise SystemExit("Input CSV has no header.")

        if args.lemma_col not in reader.fieldnames:
            raise SystemExit(f"Missing lemma column {args.lemma_col!r}")

        group_cols = args.group_cols or infer_group_cols(reader.fieldnames)
        if not group_cols:
            raise SystemExit(
                "Could not infer group columns. Either groups were not appended or prob_k columns are missing. "
                "Provide --group-cols explicitly."
            )

        missing = [g for g in group_cols if g not in reader.fieldnames]
        if missing:
            raise SystemExit(f"Missing group columns in input: {missing}")

        sums: Dict[str, Dict[str, float]] = {g: defaultdict(float) for g in group_cols}
        counts: Dict[str, int] = defaultdict(int)

        for row in reader:
            lemma = (row.get(args.lemma_col) or "").strip()
            if not lemma:
                continue

            counts[lemma] += 1
            for g in group_cols:
                val = (row.get(g) or "").strip()
                if not val:
                    continue
                try:
                    sums[g][lemma] += float(val)
                except ValueError:
                    pass

    # Compute means[group][lemma]
    means: Dict[str, Dict[str, float]] = {g: {} for g in group_cols}
    for lemma, c in counts.items():
        for g in group_cols:
            means[g][lemma] = (sums[g][lemma] / c) if c else 0.0

    # Write CSV
    out_fields = [args.lemma_col] + group_cols + ["count"]
    with open(output_csv, "w", newline="", encoding=args.encoding) as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()
        for lemma in sorted(counts):
            c = counts[lemma]
            row = {args.lemma_col: lemma, "count": c}
            for g in group_cols:
                row[g] = f"{means[g][lemma]:.10g}"
            writer.writerow(row)

    print(f"Wrote {output_csv}")

    # Write Excel if requested
    if excel_path:
        write_excel(
            xlsx_path=excel_path,
            lemma_col=args.lemma_col,
            group_cols=group_cols,
            counts=counts,
            means=means,
        )
        print(f"Wrote {excel_path}")


if __name__ == "__main__":
    main()