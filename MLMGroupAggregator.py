#!/usr/bin/env python3
"""
Recompute group probabilities from an existing MLM-output CSV using a NEW group file.

Use-case:
  - You already ran roberta_mlm_on_verbs.py and have a CSV that includes:
      token_1, prob_1, ..., token_k, prob_k
  - You now have a new grouping CSV (columns=groups, cells=lemmas; header row=group names)
  - You want to compute new per-row group probabilities by summing probs of tokens whose
    (lemmatized) form belongs to each group, WITHOUT re-running the model.

Output modes:
  1) Default: write ALL original columns + appended group columns.
  2) --short: write only lemma + group columns (optionally also count via --include-count).

Notes:
  - Lemmatization uses lemminflect.getLemma(upos="VERB") matching your MLM script.
  - Group file assumed to contain verb lemmas; we normalize group lemmas to lowercase.
  - top_k is inferred from the header by finding the largest prob_N present (or set via --top-k).

Requirements:
  pip install lemminflect

Examples:
  python recompute_groups_from_mlm_csv.py mlm_out.csv new_groups.csv mlm_out_regrouped.csv
  python recompute_groups_from_mlm_csv.py mlm_out.csv new_groups.csv regrouped_short.csv --short
  python recompute_groups_from_mlm_csv.py mlm_out.csv new_groups.csv regrouped_short.csv --short --include-count
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from typing import Dict, List, Set, Tuple

from lemminflect import getLemma


PROB_COL_RE = re.compile(r"^prob_(\d+)$")


# --------------------------- normalization ---------------------------

def normalize_pred_token(tok: str) -> str:
    """Normalize predicted token for matching against group lemmas."""
    t = (tok or "").strip().lower()
    if not t:
        return t
    if all(not ch.isalnum() for ch in t):
        return t
    lemmas = getLemma(t, upos="VERB") or getLemma(t)
    if lemmas:
        return lemmas[0].lower()
    return t


# --------------------------- group loading ---------------------------

def load_groups_csv(path: str, encoding: str, logger: logging.Logger) -> Tuple[List[str], Dict[str, Set[str]]]:
    """
    Group CSV format:
      - header row: group labels
      - cells: lemmas (may be blank)
    Returns:
      - group_labels in order
      - lemma_to_groups: normalized_lemma -> set({group_label,...})
    """
    with open(path, newline="", encoding=encoding) as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            raise ValueError(f"Group CSV {path!r} is empty.")

        group_labels = [h.strip() for h in headers if h.strip()]
        if not group_labels:
            raise ValueError(f"Group CSV {path!r} has no non-empty header labels.")

        lemma_to_groups: Dict[str, Set[str]] = {}

        for row in reader:
            row = (row + [""] * len(headers))[:len(headers)]
            for col_idx, cell in enumerate(row):
                lemma = (cell or "").strip()
                if not lemma:
                    continue
                group = (headers[col_idx] or "").strip()
                if not group:
                    continue
                key = lemma.strip().lower()
                lemma_to_groups.setdefault(key, set()).add(group)

        logger.info(
            f"Loaded groups from {path}: {len(group_labels)} groups, {len(lemma_to_groups)} unique lemmas"
        )
        return group_labels, lemma_to_groups


# --------------------------- top_k inference ---------------------------

def infer_top_k(fieldnames: List[str]) -> int:
    """Infer top_k from columns prob_1..prob_k (take the maximum N)."""
    k = 0
    for name in fieldnames:
        m = PROB_COL_RE.match((name or "").strip())
        if m:
            k = max(k, int(m.group(1)))
    return k


def build_group_output_columns(
    input_fieldnames: List[str],
    group_labels: List[str],
) -> Tuple[List[str], Dict[str, str]]:
    """
    Returns:
      - group_out_cols in output order
      - group_colname_map: group_label -> actual output column name (disambiguated)
    """
    existing = set(input_fieldnames)
    out_cols: List[str] = []
    mapping: Dict[str, str] = {}

    for g in group_labels:
        colname = g
        if colname in existing:
            colname = f"{g}_group"
        suffix = 2
        while colname in existing:
            colname = f"{g}_group{suffix}"
            suffix += 1
        mapping[g] = colname
        out_cols.append(colname)
        existing.add(colname)

    return out_cols, mapping


# --------------------------- main ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mlm_csv", help="Existing MLM output CSV containing token_i/prob_i columns")
    ap.add_argument("group_csv", help="New group CSV (columns=groups, header row=group labels)")
    ap.add_argument("output_csv", help="Output CSV with recomputed group columns appended")
    ap.add_argument("--top-k", type=int, default=0, help="Use this top_k instead of inferring from header")
    ap.add_argument("--lemma-col", default="lemma", help="Lemma column in the MLM CSV (default: lemma)")
    ap.add_argument(
        "--short",
        action="store_true",
        help="Output only lemma and group probabilities (instead of all original columns).",
    )
    ap.add_argument(
        "--include-count",
        action="store_true",
        help="With --short, also include a per-lemma count column (count within this file).",
    )
    ap.add_argument("--encoding", default="utf-8", help="CSV encoding (default: utf-8)")
    ap.add_argument("--log-every", type=int, default=100000, help="Log progress every N written rows")
    ap.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("regroup")

    group_labels, lemma_to_groups = load_groups_csv(args.group_csv, args.encoding, logger)

    processed = 0
    skipped = 0
    lemma_counts: Dict[str, int] = {}  # only used if --short --include-count

    with open(args.mlm_csv, newline="", encoding=args.encoding) as fin, \
         open(args.output_csv, "w", newline="", encoding=args.encoding) as fout:

        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise SystemExit("Input MLM CSV has no header.")
        fieldnames = reader.fieldnames

        if args.lemma_col not in fieldnames:
            raise SystemExit(f"Input MLM CSV missing lemma column {args.lemma_col!r}")

        k = args.top_k if args.top_k and args.top_k > 0 else infer_top_k(fieldnames)
        if k <= 0:
            raise SystemExit("Could not infer top_k from header; provide --top-k explicitly.")

        # verify required token/prob columns exist for 1..k
        missing_cols = []
        for i in range(1, k + 1):
            if f"token_{i}" not in fieldnames:
                missing_cols.append(f"token_{i}")
            if f"prob_{i}" not in fieldnames:
                missing_cols.append(f"prob_{i}")
        if missing_cols:
            raise SystemExit(
                f"Missing required token/prob columns for top_k={k}: "
                f"{missing_cols[:10]}{' ...' if len(missing_cols) > 10 else ''}"
            )

        # build output columns
        group_out_cols, group_colname_map = build_group_output_columns(fieldnames, group_labels)

        if args.short:
            out_fieldnames = [args.lemma_col] + group_out_cols
            if args.include_count:
                out_fieldnames.append("lemma_count")
        else:
            out_fieldnames = list(fieldnames) + group_out_cols

        writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
        writer.writeheader()

        for row in reader:
            try:
                lemma = (row.get(args.lemma_col) or "").strip()
                if not lemma:
                    raise ValueError(f"Empty lemma in column {args.lemma_col!r}")

                # initialize sums to 0 for all groups
                group_sums: Dict[str, float] = {g: 0.0 for g in group_labels}

                # accumulate over stored top_k predictions
                for i in range(1, k + 1):
                    tok = row.get(f"token_{i}", "")
                    p_str = row.get(f"prob_{i}", "")
                    if not tok or not p_str:
                        continue
                    try:
                        prob = float(p_str)
                    except ValueError:
                        continue

                    key = normalize_pred_token(tok)
                    gs = lemma_to_groups.get(key)
                    if not gs:
                        continue
                    for g in gs:
                        group_sums[g] += prob

                if args.short:
                    out_row = {args.lemma_col: lemma}
                    for g in group_labels:
                        out_row[group_colname_map[g]] = f"{group_sums[g]:.10g}"
                    if args.include_count:
                        lemma_counts[lemma] = lemma_counts.get(lemma, 0) + 1
                        out_row["lemma_count"] = str(lemma_counts[lemma])
                else:
                    out_row = dict(row)
                    for g in group_labels:
                        out_row[group_colname_map[g]] = f"{group_sums[g]:.10g}"

                writer.writerow(out_row)
                processed += 1

            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    logger.warning(f"Skipping row due to error: {e}")
                continue

            if args.log_every > 0 and processed % args.log_every == 0:
                logger.info(f"Written: {processed:,} | skipped: {skipped:,}")

    logger.info(f"Done. Written={processed:,} skipped={skipped:,} output={args.output_csv}")


if __name__ == "__main__":
    main()