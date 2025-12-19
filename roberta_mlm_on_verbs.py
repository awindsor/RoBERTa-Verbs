#!/usr/bin/env python3
"""
Run RoBERTa masked-LM inference on a "verbs file" CSV.

For each row:
  - read sentence + span_in_sentence_char (format "start:end")
  - replace that character span with RoBERTa's mask token (<mask>)
  - run masked language model inference
  - write the original row plus 2*top_k new columns:
      token_1, prob_1, token_2, prob_2, ..., token_k, prob_k

Optional grouping:
  - provide a group CSV whose columns are groups and whose cells are verb lemmas
    (first row is the header = group labels).
  - after getting top_k predictions for a row, sum probabilities of predicted tokens
    that appear in each group; append those group probability columns to the output.

Streaming-friendly:
  - reads input CSV line-by-line
  - writes output CSV line-by-line
  - holds only a small batch in memory

Requirements:
  pip install transformers torch

Example:
  python roberta_mlm_on_verbs.py verbs.csv verbs_with_mlm.csv --model roberta-base --batch-size 16 --top-k 10

With groups:
  python roberta_mlm_on_verbs.py verbs.csv verbs_with_mlm.csv --group-csv verb_groups.csv

Debug:
  python roberta_mlm_on_verbs.py verbs.csv out.csv --log-level DEBUG --debug-limit 100
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from lemminflect import getLemma



# --------------------------- helpers ---------------------------

def parse_span(span_str: str) -> Tuple[int, int]:
    # expected "start:end"
    a, b = span_str.split(":")
    start = int(a)
    end = int(b)
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"Invalid span: {span_str}")
    return start, end


def mask_sentence(sentence: str, span_str: str, mask_token: str) -> str:
    start, end = parse_span(span_str)
    if end > len(sentence):
        raise ValueError(f"Span {span_str} out of bounds for sentence length {len(sentence)}")
    return sentence[:start] + mask_token + sentence[end:]


def decode_token(tokenizer, token_id: int) -> str:
    # For RoBERTa, decoding a single token id may include a leading space.
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False).strip()


def normalize_pred_token(tok: str) -> str:
    """
    Normalize a predicted token for matching against group-lemma lists.

    Steps:
      1) basic cleanup + lowercase
      2) lemmatize via lemminflect (best-effort)
         - if lemminflect returns multiple lemmas, take the first
         - if it returns nothing, fall back to the cleaned token
    """
    t = tok.strip().lower()

    # Optional: fast path for empty / punctuation-only
    if not t or all(not ch.isalnum() for ch in t):
        return t

    # lemminflect expects a surface form and returns a tuple of candidate lemmas
    # e.g., getLemma("running") -> ("run",)
    lemmas = getLemma(t,upos="VERB")
    if lemmas:
        return lemmas[0].lower()

    return t

def load_groups_csv(path: str, encoding: str, logger: logging.Logger) -> Tuple[List[str], Dict[str, Set[str]]]:
    """
    Reads a group CSV where:
      - first row contains group labels (headers)
      - each subsequent row contains lemmas in columns (may have blanks)
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

        group_labels = [h.strip() for h in headers if h.strip() != ""]
        if not group_labels:
            raise ValueError(f"Group CSV {path!r} has no non-empty header labels.")

        lemma_to_groups: Dict[str, Set[str]] = {}

        for row_idx, row in enumerate(reader, start=2):
            # pad/trim to header length for safety
            row = (row + [""] * len(headers))[:len(headers)]
            for col_idx, cell in enumerate(row):
                lemma = cell.strip()
                if not lemma:
                    continue
                if col_idx >= len(headers):
                    continue
                group = headers[col_idx].strip()
                if not group:
                    continue
                key = normalize_pred_token(lemma)
                lemma_to_groups.setdefault(key, set()).add(group)

        logger.info(
            f"Loaded groups from {path}: {len(group_labels)} groups, {len(lemma_to_groups)} unique lemmas"
        )
        return group_labels, lemma_to_groups


def topk_for_batch(
    model,
    tokenizer,
    texts: List[str],
    top_k: int,
    device: torch.device,
) -> List[List[Tuple[str, float]]]:
    """
    Returns per-text list of (decoded_token, probability) length top_k.
    Assumes each text contains exactly one mask token.
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        out = model(**enc)
        logits = out.logits  # [B, T, V]

    mask_id = tokenizer.mask_token_id
    input_ids = enc["input_ids"]  # [B, T]
    mask_positions = (input_ids == mask_id).nonzero(as_tuple=False)  # [N, 2] with (b, t)

    B = input_ids.size(0)
    mask_index: List[Optional[int]] = [None] * B
    for b, t in mask_positions.tolist():
        if mask_index[b] is None:
            mask_index[b] = t

    results: List[List[Tuple[str, float]]] = []
    for b in range(B):
        t = mask_index[b]
        if t is None:
            results.append([("", 0.0)] * top_k)
            continue

        vocab_logits = logits[b, t, :]  # [V]
        probs = torch.softmax(vocab_logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=top_k)

        row: List[Tuple[str, float]] = []
        for pid, p in zip(top_ids.tolist(), top_probs.tolist()):
            row.append((decode_token(tokenizer, pid), float(p)))
        results.append(row)

    return results


@dataclass
class PendingRow:
    row: Dict[str, str]
    masked_text: str


# --------------------------- main ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", help="Verbs file CSV (from your extractor)")
    ap.add_argument("output_csv", help="Output CSV with MLM predictions appended")
    ap.add_argument("--model", default="roberta-base", help="HF model name (default: roberta-base)")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    ap.add_argument("--top-k", type=int, default=10, help="Top-k predictions to write (default: 10)")
    ap.add_argument("--group-csv", default=None, help="Optional CSV of lemma groups (columns), header row = group labels")
    ap.add_argument("--device", default=None, help='Device: "cpu", "cuda", "mps", or leave blank for auto')
    ap.add_argument("--log-every", type=int, default=1000, help="Log progress every N rows (default: 1000)")
    ap.add_argument("--log-level", default="INFO", help='Logging level: DEBUG, INFO, WARNING (default: INFO)')
    ap.add_argument("--debug-limit", type=int, default=0,
                    help="If >0, stop after this many rows seen (processed+skipped). Handy for debugging (e.g., 100).")
    ap.add_argument("--encoding", default="utf-8", help="CSV encoding (default: utf-8)")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("mlm")

    if args.top_k <= 0:
        raise SystemExit("--top-k must be a positive integer.")

    # Device selection
    if args.device:
        dev = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")

    # Optional groups
    group_labels: List[str] = []
    lemma_to_groups: Dict[str, Set[str]] = {}
    if args.group_csv:
        try:
            group_labels, lemma_to_groups = load_groups_csv(args.group_csv, args.encoding, logger)
        except Exception as e:
            raise SystemExit(f"Failed to load --group-csv {args.group_csv!r}: {e}")

    logger.info(f"Loading model/tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.to(dev)
    model.eval()

    if tokenizer.mask_token is None:
        raise SystemExit("Tokenizer has no mask_token; this script requires a masked LM tokenizer.")

    mask_token = tokenizer.mask_token  # RoBERTa: "<mask>"
    logger.info(
        f"Using device={dev} mask_token={mask_token!r} top_k={args.top_k} "
        f"batch_size={args.batch_size} debug_limit={args.debug_limit or 'OFF'}"
    )

    # New columns: token_1, prob_1, ..., token_k, prob_k
    pred_cols: List[str] = []
    for i in range(1, args.top_k + 1):
        pred_cols.extend([f"token_{i}", f"prob_{i}"])

    needed_cols = {"sentence", "span_in_sentence_char"}
    processed = 0
    skipped = 0

    pending: List[PendingRow] = []

    with open(args.input_csv, newline="", encoding=args.encoding) as fin, \
         open(args.output_csv, "w", newline="", encoding=args.encoding) as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise SystemExit("Input CSV has no header.")

        missing = needed_cols - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"Input CSV missing required columns: {sorted(missing)}")

        # Group output columns: include group names, but avoid collisions with existing columns.
        group_out_cols: List[str] = []
        group_colname_map: Dict[str, str] = {}  # group_label -> actual_output_column_name
        if group_labels:
            existing = set(reader.fieldnames) | set(pred_cols)
            for g in group_labels:
                colname = g
                if colname in existing:
                    # auto-disambiguate
                    colname = f"{g}_group"
                    logger.warning(f"Group name {g!r} collides with existing column; writing as {colname!r}")
                group_colname_map[g] = colname
                group_out_cols.append(colname)
                existing.add(colname)

        out_fieldnames = list(reader.fieldnames) + pred_cols + group_out_cols
        writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
        writer.writeheader()

        def flush_batch(batch: List[PendingRow]) -> None:
            nonlocal processed, skipped
            texts = [pr.masked_text for pr in batch]

            if logger.isEnabledFor(logging.DEBUG):
                for i, t in enumerate(texts[: min(len(texts), 5)], start=1):
                    logger.debug(f"Masked[{i}/{len(texts)}]: {t}")

            preds = topk_for_batch(model, tokenizer, texts, args.top_k, dev)

            for pr, pr_preds in zip(batch, preds):
                out_row = dict(pr.row)

                # write top-k token/prob
                for j, (tok, prob) in enumerate(pr_preds, start=1):
                    out_row[f"token_{j}"] = tok
                    out_row[f"prob_{j}"] = f"{prob:.10g}"

                # if groups enabled: sum probabilities of predicted tokens that match any group lemma
                if group_labels:
                    group_sums: Dict[str, float] = {g: 0.0 for g in group_labels}
                    for tok, prob in pr_preds:
                        key = normalize_pred_token(tok)
                        gs = lemma_to_groups.get(key)
                        if not gs:
                            continue
                        for g in gs:
                            group_sums[g] += prob

                    for g in group_labels:
                        out_row[group_colname_map[g]] = f"{group_sums[g]:.10g}"

                writer.writerow(out_row)
                processed += 1

        for row_idx, row in enumerate(reader, start=1):
            if args.debug_limit and (processed + skipped) >= args.debug_limit:
                logger.info(f"Debug limit reached ({args.debug_limit}); stopping early.")
                break

            try:
                raw_sent = row["sentence"]
                span = row["span_in_sentence_char"]

                masked_raw = mask_sentence(raw_sent, span, mask_token)
                # Keep your earlier leading-space behavior (often helps RoBERTa) but preserve for debugging
                masked = " " + masked_raw.strip()

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Row {row_idx}: span={span} raw={raw_sent!r}")
                    logger.debug(f"Row {row_idx}: masked={masked!r}")

                pending.append(PendingRow(row=row, masked_text=masked))

            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    logger.warning(f"Skipping row {row_idx} due to error: {e}")
                continue

            if len(pending) >= args.batch_size:
                flush_batch(pending)
                pending.clear()

            if (processed + skipped) % args.log_every == 0 and (processed + skipped) > 0:
                logger.info(f"Rows seen: {processed + skipped:,} | written: {processed:,} | skipped: {skipped:,}")

        if pending and (not args.debug_limit or (processed + skipped) < args.debug_limit):
            flush_batch(pending)
            pending.clear()

    logger.info(f"Done. Written={processed:,} skipped={skipped:,} output={args.output_csv}")


if __name__ == "__main__":
    main()