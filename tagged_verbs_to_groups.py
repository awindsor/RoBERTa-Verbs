#!/usr/bin/env python3
"""
Input CSV format:
  header: lemma,frequency,<group1>,<group2>,...,<groupK>
  rows:   lemma,<int>, (blank or 0 or 1) for each group column
          with at most one '1' across the group columns.

Output CSV format:
  header: <group1>,<group2>,...,<groupK>
  rows:   lemmas listed under the group they were tagged with (1),
          blank elsewhere. (Ragged columns are padded with blanks.)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import logging 



def read_and_group(input_path: str, ignore = None) -> Tuple[List[str], Dict[str, List[str]]]:
    if not ignore:
        ignore = []
    groups_to_lemmas: Dict[str, List[str]] = defaultdict(list)

    with open(input_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError("Input CSV is empty.")

        # Find lemma + frequency columns (required), then everything after is group labels.
        # Expecting: lemma, frequency, group1, group2, ...
        if len(header) < 3:
            raise ValueError(
                "Expected at least 3 columns: lemma, frequency, and at least one group label."
            )

        # Be forgiving about case/whitespace in the first two headers.
        h0 = header[0].strip().lower()
        h1 = header[1].strip().lower()
        if h0 != "lemma" or h1 not in ("frequency", "freq"):
            raise ValueError(
                f"Expected first two headers to be 'lemma' and 'frequency' (or 'freq'). Got: {header[:2]}"
            )

        # Build a case-insensitive ignore set
        ignore_set = {s.strip().lower() for s in ignore} if ignore else set()

        # All group headers (columns after lemma and frequency)
        total_group_cols = max(0, len(header) - 2)
        all_group_headers = [h.strip() for h in header[2:]]

        # Determine which group-column indices to keep (relative indices 0..total_group_cols-1)
        # active_cols is a list of (relative_index, label) for non-ignored, non-empty labels
        active_cols = [
            (idx, lbl)
            for idx, lbl in enumerate(all_group_headers)
            if lbl and lbl.lower() not in ignore_set
        ]

        group_labels = [label for (_idx, label) in active_cols]
        if not group_labels:
            raise ValueError("No group labels found after the frequency column (after applying ignore).")
        logger.info(f"Found group labels: {group_labels}")

        # Initialize keys so empty groups still appear in output
        for g in group_labels:
            groups_to_lemmas[g] = []

        line_no = 1  # header is line 1
        for row in reader:
            line_no += 1
            if not row or all(cell.strip() == "" for cell in row):
                continue  # skip blank lines

            if len(row) < 2:
                print(f"Warning: line {line_no}: too few columns; skipping.", file=sys.stderr)
                continue

            # Pad/truncate so we can index all original group columns by relative position
            if len(row) < 2 + total_group_cols:
                row = row + [""] * (2 + total_group_cols - len(row))
            else:
                row = row[: 2 + total_group_cols]

            # Rebuild the row to contain only: lemma, frequency, and the non-ignored group columns (in order)
            # This preserves the existing downstream logic that expects row[2:] to align with group_labels.
            row = row[:2] + [row[2 + idx] for idx, _ in active_cols]

            lemma = row[0].strip()
            if not lemma:
                print(f"Warning: line {line_no}: empty lemma; skipping.", file=sys.stderr)
                continue

            # Validate frequency is an int (as you specified), but we don't use it in output.
            freq_str = row[1].strip()
            try:
                int(freq_str)
            except ValueError:
                print(
                    f"Warning: line {line_no}: frequency '{freq_str}' is not an integer; skipping.",
                    file=sys.stderr,
                )
                continue

            flags = row[2:]
            ones = []
            for g, v in zip(group_labels, flags):
                v = v.strip()
                if v == "":
                    continue
                if v == "1":
                    ones.append(g)
                elif v == "0" or v== "#N/A":
                    pass
                else:
                    print(
                        f"Warning: line {line_no}: invalid flag '{v}' under group '{g}' (expected blank/0/1).",
                        file=sys.stderr,
                    )

            if len(ones) > 1:
                print(
                    f"Warning: line {line_no}: multiple 1s {ones}; skipping row (lemma='{lemma}').",
                    file=sys.stderr,
                )
                continue

            if len(ones) == 1:
                groups_to_lemmas[ones[0]].append(lemma)
            # else: no group selected; ignore

    return group_labels, groups_to_lemmas


def write_columnar_output(output_path: str, group_labels: List[str], groups_to_lemmas: Dict[str, List[str]]) -> None:
    columns = [groups_to_lemmas[g] for g in group_labels]
    max_len = max((len(col) for col in columns), default=0)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(group_labels)

        for i in range(max_len):
            row = []
            for col in columns:
                row.append(col[i] if i < len(col) else "")
            writer.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Reformat a lemma-frequency-groupflag CSV into a columnar CSV listing lemmas under each group."
    )
    ap.add_argument("input_csv", help="Path to input CSV (lemma, frequency, group columns).")
    ap.add_argument("output_csv", help="Path to output CSV (group columns with lemmas listed underneath).")
    ap.add_argument("--ignore", nargs="*", help="List of group labels to ignore ")
    args = ap.parse_args()

    try:
        group_labels, groups_to_lemmas = read_and_group(args.input_csv,ignore=args.ignore)
        write_columnar_output(args.output_csv, group_labels, groups_to_lemmas)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    raise SystemExit(main())