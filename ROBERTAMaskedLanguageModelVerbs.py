#!/usr/bin/env python3
"""
Run RoBERTa masked-LM inference on a "verbs file" CSV.

For each row:
  - read sentence + span_in_sentence_char (format "start:end")
  - replace that character span with RoBERTa's mask token (<mask>)
  - run masked language model inference
  - write the original row plus 2*top_k new columns:
      token_1, prob_1, token_2, prob_2, ..., token_k, prob_k

Streaming-friendly:
  - reads input CSV line-by-line
  - writes output CSV line-by-line
  - holds only a small batch in memory

Requirements:
  pip install transformers torch lemminflect

Example:
  python roberta_mlm_on_verbs.py verbs.csv verbs_with_mlm.csv --model roberta-base --batch-size 16 --top-k 10

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
    seen = 0
    next_log_at = args.log_every

    pending: List[PendingRow] = []

    def maybe_log_progress(force: bool = False) -> None:
        nonlocal next_log_at
        if args.log_every <= 0:
            return
        if force:
            logger.info(f"Rows seen: {seen:,} | written: {processed:,} | skipped: {skipped:,}")
            return
        while seen >= next_log_at and next_log_at > 0:
            logger.info(f"Rows seen: {seen:,} | written: {processed:,} | skipped: {skipped:,}")
            next_log_at += args.log_every

    with open(args.input_csv, newline="", encoding=args.encoding) as fin, \
         open(args.output_csv, "w", newline="", encoding=args.encoding) as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise SystemExit("Input CSV has no header.")

        missing = needed_cols - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"Input CSV missing required columns: {sorted(missing)}")

        out_fieldnames = list(reader.fieldnames) + pred_cols
        writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
        writer.writeheader()

        def flush_batch(batch: List[PendingRow]) -> None:
            nonlocal processed
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

                writer.writerow(out_row)
                processed += 1

        for row_idx, row in enumerate(reader, start=1):
            seen += 1

            if args.debug_limit and (processed + skipped) >= args.debug_limit:
                logger.info(f"Debug limit reached ({args.debug_limit}); stopping early.")
                break

            try:
                raw_sent = row["sentence"]
                span = row["span_in_sentence_char"]

                masked_raw = mask_sentence(raw_sent, span, mask_token)
                masked = " " + masked_raw.strip()

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Row {row_idx}: span={span} raw={raw_sent!r}")
                    logger.debug(f"Row {row_idx}: masked={masked!r}")

                pending.append(PendingRow(row=row, masked_text=masked))

            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    logger.warning(f"Skipping row {row_idx} due to error: {e}")
                maybe_log_progress()
                continue

            if len(pending) >= args.batch_size:
                flush_batch(pending)
                pending.clear()
                maybe_log_progress()

            else:
                maybe_log_progress()

        if pending and (not args.debug_limit or (processed + skipped) < args.debug_limit):
            flush_batch(pending)
            pending.clear()

        maybe_log_progress(force=True)

    logger.info(f"Done. Written={processed:,} skipped={skipped:,} output={args.output_csv}")


if __name__ == "__main__":
    main()