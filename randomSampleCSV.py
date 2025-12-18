#!/usr/bin/env python3
"""
Randomly sample n rows from a CSV without loading the whole file into memory.

- Two passes:
  1) Count rows (excluding header)
  2) Select n distinct row indices and write those rows

Usage:
  python sample_csv.py input.csv output.csv 100
"""

import argparse
import csv
import random


def count_data_rows(path: str) -> int:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        # header
        next(reader, None)
        return sum(1 for _ in reader)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("output_csv")
    ap.add_argument("n", type=int, help="Sample size")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    args = ap.parse_args()

    if args.n < 0:
        raise SystemExit("n must be >= 0")

    if args.seed is not None:
        random.seed(args.seed)

    total = count_data_rows(args.input_csv)
    if args.n > total:
        raise SystemExit(f"Requested n={args.n} but file has only {total} data rows.")

    # Choose which row numbers (0-based over data rows) to keep
    keep = set(random.sample(range(total), args.n))

    with open(args.input_csv, newline="", encoding="utf-8") as fin, \
         open(args.output_csv, "w", newline="", encoding="utf-8") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        header = next(reader, None)
        if header is None:
            raise SystemExit("Input CSV is empty.")
        writer.writerow(header)

        for i, row in enumerate(reader):  # i is 0-based over data rows
            if i in keep:
                writer.writerow(row)

    print(f"Wrote {args.n} sampled rows to {args.output_csv}")


if __name__ == "__main__":
    main()