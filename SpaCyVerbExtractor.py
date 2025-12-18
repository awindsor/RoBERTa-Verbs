#!/usr/bin/env python3
"""
Chunked verb extraction with spaCy + basic logging/progress reporting.

For each input document:
  - stream overlapping character chunks
  - sentence-split each chunk with spaCy
  - for each sentence, iterate verb tokens (POS == VERB; optionally include AUX)
  - output one row per verb with:
      lemma, surface_lower, span_in_sentence_char, sentence_text
  - de-duplicate sentences that reappear due to chunk overlap

Output columns:
  doc_path, chunk_start_char, sent_start_char_in_doc, sent_index_in_doc_approx,
  token_index_in_sent, lemma, surface_lower, span_in_sentence_char, sentence

Requirements:
  pip install spacy
  python -m spacy download en_core_web_sm

Examples:
  python extract_verbs_chunked_logged.py big.txt -o verbs.csv
  python extract_verbs_chunked_logged.py --paths-file paths.txt -o verbs.tsv --tsv --include-aux
  python extract_verbs_chunked_logged.py data/*.txt --chunk-size 1000000 --overlap 10000 --log-level INFO
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import spacy


def iter_paths(cli_paths: List[str], paths_file: Optional[str]) -> List[Path]:
    paths: List[Path] = [Path(p) for p in cli_paths]
    if paths_file:
        pf = Path(paths_file)
        for line in pf.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(Path(line))

    # Deduplicate while preserving order
    seen = set()
    out: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def stream_char_chunks(
    path: Path,
    *,
    encoding: str,
    chunk_size: int,
    overlap: int,
) -> Generator[Tuple[int, str], None, None]:
    """
    Yields (chunk_start_char_in_doc, chunk_text) with overlap.
    chunk_size and overlap are in characters (not bytes).

    FIX: Do not emit a final "tail-only" chunk at EOF.
    """
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    with path.open("r", encoding=encoding, errors="replace") as f:
        start_char = 0
        prev_tail = ""

        while True:
            to_read = chunk_size - len(prev_tail)
            if to_read <= 0:
                prev_tail = prev_tail[-(chunk_size - 1):]
                to_read = chunk_size - len(prev_tail)

            block = f.read(to_read)

            # If no new data, we're done. (Prevents tail-only chunk.)
            if not block:
                break

            chunk_start = start_char - len(prev_tail)
            chunk_text = prev_tail + block
            yield (chunk_start, chunk_text)

            prev_tail = chunk_text[-overlap:] if overlap else ""
            start_char += len(block)


def setup_logging(level_name: str) -> logging.Logger:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("extract_verbs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="Paths to text files.")
    ap.add_argument("--paths-file", help="Text file with one path per line (# comments allowed).")
    ap.add_argument("-o", "--output", default="verbs.csv", help="Output file path (default: verbs.csv).")
    ap.add_argument("--tsv", action="store_true", help="Write TSV instead of CSV.")
    ap.add_argument("--model", default="en_core_web_sm", help="spaCy model name (default: en_core_web_sm).")
    ap.add_argument("--encoding", default="utf-8", help="Input file encoding (default: utf-8).")
    ap.add_argument("--include-aux", action="store_true", help="Also treat AUX tokens as verbs.")
    ap.add_argument("--chunk-size", type=int, default=2_000_000, help="Chunk size in characters (default: 2,000,000).")
    ap.add_argument("--overlap", type=int, default=5_000, help="Chunk overlap in characters (default: 5,000).")
    ap.add_argument(
        "--dedupe-window",
        type=int,
        default=50_000,
        help="How many recent sentences to remember for de-duplication (default: 50,000).",
    )
    ap.add_argument(
        "--heartbeat-chunks",
        type=int,
        default=10,
        help="Log a progress heartbeat every N chunks (default: 10).",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = ap.parse_args()

    logger = setup_logging(args.log_level)

    paths = iter_paths(args.paths, args.paths_file)
    if not paths:
        raise SystemExit("No input paths provided.")

    # Load spaCy with only what we need.
    # We keep tagger/lemmatizer; disable NER to save time/memory.
    t0 = time.time()
    logger.info(f"Loading spaCy model: {args.model}")
    nlp = spacy.load(args.model, disable=["ner", "textcat"])
    if "parser" not in nlp.pipe_names:
        # Parser provides sentence boundaries; if absent, use rule-based sentencizer.
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
            logger.info("Parser not found in pipeline; added sentencizer for sentence boundaries.")
    logger.info(f"spaCy pipeline: {nlp.pipe_names} (loaded in {time.time() - t0:.1f}s)")

    out_path = Path(args.output)
    delimiter = "\t" if args.tsv else ","

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow([
            "doc_path",
            "chunk_start_char",
            "sent_start_char_in_doc",
            "sent_index_in_doc_approx",
            "token_index_in_sent",
            "lemma",
            "surface_lower",
            "span_in_sentence_char",
            "sentence",
        ])

        overall_docs = 0
        overall_chunks = 0
        overall_sents = 0
        overall_verbs = 0
        overall_start = time.time()

        for doc_path in paths:
            overall_docs += 1
            if not doc_path.exists():
                logger.warning(f"Skipping missing path: {doc_path}")
                continue

            logger.info(f"Starting document: {doc_path}")

            # NOTE: st_size is bytes, while chunk indices are characters.
            # We use it only as a rough progress proxy.
            try:
                file_size_bytes = doc_path.stat().st_size
            except OSError:
                file_size_bytes = 0

            doc_start = time.time()

            # De-dupe structures:
            # key = (sentence_start_char_in_doc, sentence_text) -> insertion counter
            seen: Dict[Tuple[int, str], int] = {}
            order: List[Tuple[int, str]] = []
            sent_counter = 0
            verb_counter = 0
            chunk_counter = 0

            for chunk_start, chunk_text in stream_char_chunks(
                doc_path,
                encoding=args.encoding,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            ):
                chunk_counter += 1
                overall_chunks += 1

                chunk_end = chunk_start + len(chunk_text)

                # Rough % progress using bytes proxy (best-effort)
                if file_size_bytes > 0:
                    pct = min(100.0, (chunk_end / file_size_bytes) * 100.0)
                else:
                    pct = 0.0

                logger.info(
                    f"  Chunk {chunk_counter:6d} | chars {chunk_start:,}–{chunk_end:,} | ~{pct:6.2f}%"
                )

                doc = nlp(chunk_text)

                for sent in doc.sents:
                    sent_start_in_doc = chunk_start + sent.start_char
                    key = (sent_start_in_doc, sent.text)

                    if key in seen:
                        continue  # duplicated via overlap

                    # record + prune occasionally
                    seen[key] = sent_counter
                    order.append(key)
                    sent_counter += 1

                    if len(order) > args.dedupe_window:
                        # drop oldest ~10% to amortize cost
                        drop_n = max(1, args.dedupe_window // 10)
                        if args.log_level == "DEBUG":
                            logger.debug(f"Pruning dedupe cache: dropping {drop_n} oldest sentences")
                        for _ in range(drop_n):
                            old = order.pop(0)
                            seen.pop(old, None)

                    sent_text = sent.text

                    for tok_i, tok in enumerate(sent):
                        is_verb = tok.pos_ == "VERB" or (args.include_aux and tok.pos_ == "AUX")
                        if not is_verb:
                            continue

                        verb_counter += 1

                        # Character span relative to the sentence
                        start_in_sent = tok.idx - sent.start_char
                        end_in_sent = start_in_sent + len(tok.text)

                        writer.writerow([
                            str(doc_path),
                            chunk_start,
                            sent_start_in_doc,
                            seen[key],  # approx sentence index (after de-dupe)
                            tok_i,
                            tok.lemma_,
                            tok.text.lower(),
                            f"{start_in_sent}:{end_in_sent}",
                            sent_text,
                        ])

                # Periodic heartbeat
                if args.heartbeat_chunks > 0 and (chunk_counter % args.heartbeat_chunks == 0):
                    elapsed = time.time() - doc_start
                    logger.info(
                        f"    Heartbeat: {sent_counter:,} sentences | {verb_counter:,} verbs | {elapsed:,.1f}s elapsed"
                    )

            elapsed_doc = time.time() - doc_start
            overall_sents += sent_counter
            overall_verbs += verb_counter

            logger.info(
                f"Finished document: {doc_path} | "
                f"{chunk_counter:,} chunks | {sent_counter:,} sentences | {verb_counter:,} verbs | "
                f"{elapsed_doc:,.1f}s"
            )

        elapsed_all = time.time() - overall_start
        logger.info(
            "Run complete | "
            f"{overall_docs:,} docs (including missing paths) | "
            f"{overall_chunks:,} chunks | {overall_sents:,} sentences | {overall_verbs:,} verbs | "
            f"{elapsed_all:,.1f}s total"
        )

    logger.info(f"Wrote output: {out_path}")


if __name__ == "__main__":
    main()