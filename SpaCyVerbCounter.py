#!/usr/bin/env python3
"""
Aggregate verb frequencies from extract_verbs CSV (generator-based).

Usage:
  python verb_freqs.py input.csv output.csv --field lemma
  python verb_freqs.py input.csv output.csv --field surface_form
"""

import csv
import argparse
from collections import Counter
from typing import Iterator


def stream_column(path: str, field: str) -> Iterator[str]:
    """Yield values from a single column, one row at a time."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row[field]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("output_csv")
    ap.add_argument(
        "--field",
        choices=["lemma", "surface_lower"],
        required=True
    )
    args = ap.parse_args()

    counter = Counter(stream_column(args.input_csv, args.field))

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([args.field, "freq"])
        for token, freq in counter.most_common():
            writer.writerow([token, freq])


if __name__ == "__main__":
    main()