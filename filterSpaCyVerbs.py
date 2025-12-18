#!/usr/bin/env python3
"""
Filter a verb CSV by frequency of a chosen field (two-pass, streaming).

Pass 1: count frequencies of the chosen field (lemma or surface_lower)
Pass 2: write only rows whose field frequency is within [min_freq, max_freq] (inclusive)

Never reads the whole file into memory (but does store the Counter of unique values).
"""

import argparse
import csv
from collections import Counter
from typing import Optional


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", help="Verb file CSV produced by the extractor")
    ap.add_argument("output_csv", help="Filtered output CSV")
    ap.add_argument("--field", choices=["lemma", "surface_lower"], required=True)
    ap.add_argument("--min-freq", type=int, default=None, help="Keep values with freq >= min_freq")
    ap.add_argument("--max-freq", type=int, default=None, help="Keep values with freq <= max_freq")
    return ap.parse_args()


def count_field_freq(path: str, field: str) -> Counter:
    counts = Counter()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or field not in reader.fieldnames:
            raise ValueError(f"Field '{field}' not found in input header: {reader.fieldnames}")
        for row in reader:
            counts[row[field]] += 1
    return counts


def in_range(freq: int, min_freq: Optional[int], max_freq: Optional[int]) -> bool:
    if min_freq is not None and freq < min_freq:
        return False
    if max_freq is not None and freq > max_freq:
        return False
    return True


def filter_rows(input_csv: str, output_csv: str, field: str, counts: Counter,
                min_freq: Optional[int], max_freq: Optional[int]) -> None:
    with open(input_csv, newline="", encoding="utf-8") as fin, open(
        output_csv, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError("Input CSV appears to have no header.")
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            val = row[field]
            if in_range(counts[val], min_freq, max_freq):
                writer.writerow(row)


def main():
    args = parse_args()

    if args.min_freq is None and args.max_freq is None:
        raise SystemExit("Error: At least one of --min-freq or --max-freq must be specified.")

    if args.min_freq is not None and args.min_freq < 1:
        raise SystemExit("Error: --min-freq must be >= 1 (or omit it).")
    if args.max_freq is not None and args.max_freq < 1:
        raise SystemExit("Error: --max-freq must be >= 1 (or omit it).")
    if args.min_freq is not None and args.max_freq is not None and args.min_freq > args.max_freq:
        raise SystemExit("Error: --min-freq cannot be greater than --max-freq.")

    # Pass 1: count
    counts = count_field_freq(args.input_csv, args.field)

    # Pass 2: filter + write
    filter_rows(args.input_csv, args.output_csv, args.field, counts, args.min_freq, args.max_freq)


if __name__ == "__main__":
    main()