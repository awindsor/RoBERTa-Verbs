#!/usr/bin/env python3
"""
Chunked verb extraction with spaCy + basic logging/progress reporting.

Supports both CLI and GUI modes:
  - Run without arguments: launches GUI (requires PySide6)
  - Run with arguments: uses CLI mode

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
  pip install PySide6  # for GUI mode only

CLI Examples:
  python SpaCyVerbExtractor.py big.txt -o verbs.csv
  python SpaCyVerbExtractor.py --paths-file paths.txt -o verbs.tsv --tsv --include-aux
  python SpaCyVerbExtractor.py data/*.txt --chunk-size 1000000 --overlap 10000 --log-level INFO

GUI Mode:
  python SpaCyVerbExtractor.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import spacy


# ============================================================================
# CORE EXTRACTION LOGIC (shared by CLI and GUI)
# ============================================================================

def compute_file_md5(path: Path) -> str:
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def save_run_metadata(
    output_path: Path,
    input_paths: List[Path],
    input_checksums: Dict[str, str],
    output_checksum: str,
    args: Dict[str, any],
    stats: Dict[str, any],
    status: str = "completed",
) -> None:
    """Save extraction metadata to JSON file alongside output.
    
    Args:
        status: "completed" or "stopped_by_user"
    """
    json_path = output_path.with_suffix(".json")
    
    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "status": status,
        "output_file": str(output_path),
        "output_checksum": output_checksum,
        "input_files": [str(p) for p in input_paths],
        "input_checksums": input_checksums,
        "settings": {
            "model": args.get("model", "en_core_web_sm"),
            "encoding": args.get("encoding", "utf-8"),
            "include_aux": args.get("include_aux", False),
            "chunk_size": args.get("chunk_size", 2_000_000),
            "overlap": args.get("overlap", 5_000),
            "dedupe_window": args.get("dedupe_window", 50_000),
            "heartbeat_chunks": args.get("heartbeat_chunks", 10),
            "output_format": "tsv" if args.get("tsv", False) else "csv",
            "csv_text_column": args.get("csv_text_column"),
            "include_csv_fields": args.get("include_csv_fields", False),
        },
        "statistics": stats,
    }
    
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_run_metadata(json_path: Path) -> Dict:
    """Load extraction metadata from JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify_input_file_checksums(input_paths: List[Path], metadata: Dict) -> Dict[str, str]:
    """
    Verify input file checksums against metadata.
    
    Returns dict with issues found:
    {
        "missing_files": [list of paths that don't exist],
        "checksum_mismatches": [list of paths with different checksums],
    }
    """
    input_checksums = metadata.get("input_checksums", {})
    issues = {"missing_files": [], "checksum_mismatches": []}
    
    for file_path in input_paths:
        file_path_str = str(file_path)
        
        # Check if this file is in the metadata
        if file_path_str in input_checksums:
            expected_checksum = input_checksums[file_path_str]
            
            if not file_path.exists():
                issues["missing_files"].append(file_path_str)
            else:
                actual_checksum = compute_file_md5(file_path)
                if actual_checksum != expected_checksum:
                    issues["checksum_mismatches"].append(file_path_str)
    
    return issues


def stream_char_chunks(
    path: Path,
    *,
    encoding: str,
    chunk_size: int,
    overlap: int,
    stop_check=None,
) -> Generator[Tuple[int, str], None, None]:
    """
    Yields (chunk_start_char_in_doc, chunk_text) with overlap.
    chunk_size and overlap are in characters (not bytes).
    
    Args:
        stop_check: Optional callable that returns True if processing should stop
    """
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    with path.open("r", encoding=encoding, errors="replace") as f:
        start_char = 0
        prev_tail = ""

        while True:
            if stop_check and stop_check():
                break
                
            to_read = chunk_size - len(prev_tail)
            if to_read <= 0:
                prev_tail = prev_tail[-(chunk_size - 1):]
                to_read = chunk_size - len(prev_tail)

            block = f.read(to_read)

            if not block:
                break

            chunk_start = start_char - len(prev_tail)
            chunk_text = prev_tail + block
            yield (chunk_start, chunk_text)

            prev_tail = chunk_text[-overlap:] if overlap else ""
            start_char += len(block)


def load_spacy_model(model_name: str, logger=None):
    """
    Load a spaCy model, downloading it if necessary.
    
    Args:
        model_name: Name of the spaCy model (e.g., 'en_core_web_sm')
        logger: Optional logger for informational messages
    
    Returns:
        The loaded spaCy model
    
    Raises:
        OSError: If model download fails
    """
    import subprocess
    
    try:
        # Try to load the model directly
        if logger:
            logger.info(f"Loading spaCy model: {model_name}")
        nlp = spacy.load(model_name)
        return nlp
    except OSError as e:
        # Model not found, try to download it
        if "Can't find model" in str(e) or "No such file or directory" in str(e):
            if logger:
                logger.info(f"Model '{model_name}' not found locally. Downloading...")
            
            # Download the model
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "spacy", "download", model_name],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode != 0:
                    raise OSError(f"Failed to download model '{model_name}': {result.stderr}")
                
                if logger:
                    logger.info(f"Downloaded model '{model_name}'")
                
                # Try to load again
                nlp = spacy.load(model_name)
                return nlp
            except subprocess.TimeoutExpired:
                raise OSError(f"Timeout downloading model '{model_name}'")
        else:
            # Some other error
            raise


def validate_csv_inputs(paths: List[Path], csv_text_column: Optional[str], logger=None) -> Tuple[bool, str]:
    """
    Validate that CSV files (if any) have the same text column.
    
    Returns:
        (is_valid, error_message)
    """
    if not csv_text_column:
        return True, ""
    
    csv_files = [p for p in paths if p.suffix.lower() == '.csv']
    
    if not csv_files:
        return True, ""
    
    # Check first CSV to get the column name
    first_csv_columns = None
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames
                
                if not columns:
                    return False, f"CSV file {csv_file} has no header row"
                
                if csv_text_column not in columns:
                    return False, f"CSV file {csv_file} does not have column '{csv_text_column}'. Available: {columns}"
                
                if first_csv_columns is None:
                    first_csv_columns = set(columns)
                elif set(columns) != first_csv_columns:
                    return False, f"CSV file {csv_file} has different columns than the first CSV"
        except Exception as e:
            return False, f"Error reading CSV file {csv_file}: {e}"
    
    return True, ""


def stream_csv_rows(
    path: Path,
    text_column: str,
    *,
    encoding: str,
    stop_check=None,
) -> Generator[Tuple[int, str, Dict[str, str]], None, None]:
    """
    Yields (row_number, text, other_fields_dict) from a CSV file.
    
    Args:
        path: Path to CSV file
        text_column: Name of the column containing text to extract verbs from
        encoding: File encoding
        stop_check: Optional callable that returns True if processing should stop
    """
    with open(path, 'r', encoding=encoding, errors='replace') as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, 1):  # Start at 1 for actual row numbers
            if stop_check and stop_check():
                break
            
            text = row.get(text_column, "")
            # Create dict of other fields (excluding the text column)
            other_fields = {k: v for k, v in row.items() if k != text_column}
            
            yield (row_number, text, other_fields)


def iter_paths(cli_paths: List[str], paths_file: Optional[str]) -> List[Path]:
    """Parse input paths from CLI arguments and/or paths file."""
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


# ============================================================================
# CLI MODE
# ============================================================================

def setup_logging(level_name: str) -> logging.Logger:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("extract_verbs")


def run_cli() -> None:
    ap = argparse.ArgumentParser(
        description="SpaCy Verb Extractor - Extract verbs from text files"
    )
    ap.add_argument("paths", nargs="*", help="Paths to text files.")
    ap.add_argument("--paths-file", help="Text file with one path per line (# comments allowed).")
    ap.add_argument("--output", default="verbs.csv", help="Output file path (default: verbs.csv).")
    ap.add_argument("--load-metadata", help="Load settings from a previous run JSON file (CLI args override loaded settings).")
    ap.add_argument("--tsv", action="store_true", help="Write TSV instead of CSV.")
    ap.add_argument("--csv-text-column", default=None, help="For CSV input files, name of the column containing text. If provided, allows CSV inputs.")
    ap.add_argument("--include-csv-fields", action="store_true", help="Include other CSV columns from input rows in output.")
    ap.add_argument("--model", default=None, help="spaCy model name (default: en_core_web_sm).")
    ap.add_argument("--encoding", default=None, help="Input file encoding (default: utf-8).")
    ap.add_argument("--include-aux", action="store_true", help="Also treat AUX tokens as verbs.")
    ap.add_argument("--chunk-size", type=int, default=None, help="Chunk size in characters (default: 2,000,000).")
    ap.add_argument("--overlap", type=int, default=None, help="Chunk overlap in characters (default: 5,000).")
    ap.add_argument(
        "--dedupe-window",
        type=int,
        default=None,
        help="How many recent sentences to remember for de-duplication (default: 50,000).",
    )
    ap.add_argument(
        "--heartbeat-chunks",
        type=int,
        default=None,
        help="Log a progress heartbeat every N chunks (default: 10).",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = ap.parse_args()

    # Load metadata if provided
    if args.load_metadata:
        metadata_path = Path(args.load_metadata)
        if not metadata_path.exists():
            raise SystemExit(f"Metadata file not found: {metadata_path}")
        
        metadata = load_run_metadata(metadata_path)
        settings = metadata.get("settings", {})
        
        # Use loaded settings as defaults, but CLI args override
        args.model = args.model or settings.get("model", "en_core_web_sm")
        args.encoding = args.encoding or settings.get("encoding", "utf-8")
        args.chunk_size = args.chunk_size or settings.get("chunk_size", 2_000_000)
        args.overlap = args.overlap or settings.get("overlap", 5_000)
        args.dedupe_window = args.dedupe_window or settings.get("dedupe_window", 50_000)
        args.heartbeat_chunks = args.heartbeat_chunks or settings.get("heartbeat_chunks", 10)
        
        if not args.include_aux:
            args.include_aux = settings.get("include_aux", False)
        if not args.tsv:
            args.tsv = (settings.get("output_format", "csv") == "tsv")
    else:
        # Use defaults if not loading metadata
        args.model = args.model or "en_core_web_sm"
        args.encoding = args.encoding or "utf-8"
        args.chunk_size = args.chunk_size or 2_000_000
        args.overlap = args.overlap or 5_000
        args.dedupe_window = args.dedupe_window or 50_000
        args.heartbeat_chunks = args.heartbeat_chunks or 10

    logger = setup_logging(args.log_level)

    paths = iter_paths(args.paths, args.paths_file)
    if not paths:
        raise SystemExit("No input paths provided.")
    
    # Validate CSV inputs if CSV text column is specified
    if args.csv_text_column:
        is_valid, error_msg = validate_csv_inputs(paths, args.csv_text_column, logger)
        if not is_valid:
            raise SystemExit(f"CSV validation error: {error_msg}")
        logger.info(f"✓ CSV inputs validated (text column: '{args.csv_text_column}')")
    
    # Verify input file checksums if metadata was loaded
    if args.load_metadata:
        logger.info("Verifying input file checksums against metadata...")
        issues = verify_input_file_checksums(paths, metadata)
        
        if issues["missing_files"]:
            logger.warning("⚠ Missing input files:")
            for f in issues["missing_files"]:
                logger.warning(f"   {f}")
        
        if issues["checksum_mismatches"]:
            logger.warning("⚠ Input files have changed (checksum mismatch):")
            for f in issues["checksum_mismatches"]:
                logger.warning(f"   {f}")
        
        if issues["missing_files"] or issues["checksum_mismatches"]:
            logger.warning("⚠ Proceeding with extraction despite file changes")
        else:
            logger.info("✓ All input files verified (checksum OK)")

    # Load spaCy with only what we need.
    t0 = time.time()
    nlp = load_spacy_model(args.model, logger)
    # Enable only components needed for verb extraction (tagger, parser, lemmatizer, etc.)
    # This is more reliable than disabling specific components
    components_to_enable = [
        c for c in ["tok2vec", "tagger", "parser", "senter", "attribute_ruler", "lemmatizer"]
        if c in nlp.pipe_names
    ]
    if components_to_enable:
        nlp.select_pipes(enable=components_to_enable)
    if "parser" not in nlp.pipe_names:
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
            logger.info("Parser not found in pipeline; added sentencizer for sentence boundaries.")
    logger.info(f"spaCy pipeline: {nlp.pipe_names} (loaded in {time.time() - t0:.1f}s)")

    out_path = Path(args.output)
    delimiter = "\t" if args.tsv else ","

    overall_docs = 0
    overall_chunks = 0
    overall_sents = 0
    overall_verbs = 0
    overall_start = time.time()
    stopped_by_user = False

    try:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=delimiter)
            
            # Build header dynamically based on input type
            header = [
                "doc_path",
                "chunk_start_char",
                "sent_start_char_in_doc",
                "sent_index_in_doc_approx",
                "token_index_in_sent",
                "lemma",
                "surface_lower",
                "span_in_sentence_char",
                "sentence",
            ]
            
            # Add CSV-specific columns if processing CSV files
            if args.csv_text_column:
                header.insert(1, "csv_row_number")  # Add after doc_path
                
                # If including CSV fields, we'll add them dynamically later
                # (we don't know the field names yet)
            
            writer.writerow(header)

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

            doc_start = time.time()
            sent_counter = 0
            verb_counter = 0
            chunk_counter = 0
            
            # Determine if this is a CSV file or text file
            is_csv = doc_path.suffix.lower() == '.csv' and args.csv_text_column
            
            if is_csv:
                # Process as CSV file with batch processing using nlp.pipe()
                seen_csv_rows: Dict[Tuple[int, str], int] = {}  # (csv_row_num, sent_text) -> sent_index
                order_csv = []
                
                # Collect all rows for batch processing
                csv_rows_list = []
                for csv_row_num, text_to_process, other_fields in stream_csv_rows(
                    doc_path,
                    text_column=args.csv_text_column,
                    encoding=args.encoding,
                ):
                    if text_to_process.strip():
                        # Create metadata dict for this row
                        metadata = {
                            "csv_row_num": csv_row_num,
                            "csv_fields": other_fields if args.include_csv_fields else None,
                        }
                        csv_rows_list.append((text_to_process, metadata))
                
                # Process all rows in batches using nlp.pipe() with metadata
                batch_size = 32
                for batch_idx in range(0, len(csv_rows_list), batch_size):
                    batch = csv_rows_list[batch_idx:batch_idx + batch_size]
                    
                    for doc in nlp.pipe(batch, as_tuples=True):
                        chunk_counter += 1
                        overall_chunks += 1
                        
                        metadata = doc.user_data
                        csv_row_num = metadata.get("csv_row_num")
                        csv_fields = metadata.get("csv_fields")
                        
                        for sent in doc.sents:
                            key = (csv_row_num, sent.text)
                            
                            if key in seen_csv_rows:
                                continue
                            
                            seen_csv_rows[key] = sent_counter
                            order_csv.append(key)
                            sent_counter += 1
                            
                            if len(order_csv) > args.dedupe_window:
                                drop_n = max(1, args.dedupe_window // 10)
                                for _ in range(drop_n):
                                    old = order_csv.pop(0)
                                    seen_csv_rows.pop(old, None)
                            
                            sent_text = sent.text
                            
                            for tok_i, tok in enumerate(sent):
                                is_verb = tok.pos_ == "VERB" or (args.include_aux and tok.pos_ == "AUX")
                                if not is_verb:
                                    continue
                                
                                verb_counter += 1
                                start_in_sent = tok.idx - sent.start_char
                                end_in_sent = start_in_sent + len(tok.text)
                                
                                row = [
                                    str(doc_path),
                                    csv_row_num,  # csv_row_number column
                                    0,  # chunk_start_char (not applicable for CSV)
                                    0,  # sent_start_char_in_doc (not applicable for CSV)
                                    seen_csv_rows[key],
                                    tok_i,
                                    tok.lemma_,
                                    tok.text.lower(),
                                    f"{start_in_sent}:{end_in_sent}",
                                    sent_text,
                                ]
                                
                                # Add CSV fields if requested
                                if args.include_csv_fields and csv_fields:
                                    row.extend(csv_fields.values())
                                
                                writer.writerow(row)
                        
                        if args.heartbeat_chunks > 0 and (chunk_counter % args.heartbeat_chunks == 0):
                            elapsed = time.time() - doc_start
                            logger.info(
                                f"    Heartbeat: {sent_counter:,} sentences | {verb_counter:,} verbs | {elapsed:,.1f}s elapsed"
                            )
            else:
                # Process as text file with batch processing using nlp.pipe()
                try:
                    file_size_bytes = doc_path.stat().st_size
                except OSError:
                    file_size_bytes = 0
                
                seen: Dict[Tuple[int, str], int] = {}
                order: List[Tuple[int, str]] = []
                
                # Collect chunks for batch processing
                chunk_batch = []
                batch_size = 32
                
                for chunk_start, chunk_text in stream_char_chunks(
                    doc_path,
                    encoding=args.encoding,
                    chunk_size=args.chunk_size,
                    overlap=args.overlap,
                ):
                    chunk_end = chunk_start + len(chunk_text)
                    
                    if file_size_bytes > 0:
                        pct = min(100.0, (chunk_end / file_size_bytes) * 100.0)
                    else:
                        pct = 0.0
                    
                    logger.info(
                        f"  Chunk {chunk_counter + len(chunk_batch) + 1:6d} | chars {chunk_start:,}–{chunk_end:,} | ~{pct:6.2f}%"
                    )
                    
                    # Store chunk with metadata for batch processing
                    metadata = {
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                        "file_size_bytes": file_size_bytes,
                    }
                    chunk_batch.append((chunk_text, metadata))
                    
                    # Process batch when it reaches batch_size
                    if len(chunk_batch) >= batch_size:
                        for doc in nlp.pipe(chunk_batch, as_tuples=True):
                            chunk_counter += 1
                            overall_chunks += 1
                            
                            metadata = doc.user_data
                            chunk_start = metadata.get("chunk_start")
                            
                            for sent in doc.sents:
                                sent_start_in_doc = chunk_start + sent.start_char
                                key = (sent_start_in_doc, sent.text)
                                
                                if key in seen:
                                    continue
                                
                                seen[key] = sent_counter
                                order.append(key)
                                sent_counter += 1
                                
                                if len(order) > args.dedupe_window:
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
                                    start_in_sent = tok.idx - sent.start_char
                                    end_in_sent = start_in_sent + len(tok.text)
                                    
                                    writer.writerow([
                                        str(doc_path),
                                        chunk_start,
                                        sent_start_in_doc,
                                        seen[key],
                                        tok_i,
                                        tok.lemma_,
                                        tok.text.lower(),
                                        f"{start_in_sent}:{end_in_sent}",
                                        sent_text,
                                    ])
                            
                            if args.heartbeat_chunks > 0 and (chunk_counter % args.heartbeat_chunks == 0):
                                elapsed = time.time() - doc_start
                                logger.info(
                                    f"    Heartbeat: {sent_counter:,} sentences | {verb_counter:,} verbs | {elapsed:,.1f}s elapsed"
                                )
                        
                        chunk_batch = []
                
                # Process remaining chunks in final batch
                if chunk_batch:
                    for doc in nlp.pipe(chunk_batch, as_tuples=True):
                        chunk_counter += 1
                        overall_chunks += 1
                        
                        metadata = doc.user_data
                        chunk_start = metadata.get("chunk_start")
                        
                        for sent in doc.sents:
                            sent_start_in_doc = chunk_start + sent.start_char
                            key = (sent_start_in_doc, sent.text)
                            
                            if key in seen:
                                continue
                            
                            seen[key] = sent_counter
                            order.append(key)
                            sent_counter += 1
                            
                            if len(order) > args.dedupe_window:
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
                                start_in_sent = tok.idx - sent.start_char
                                end_in_sent = start_in_sent + len(tok.text)
                                
                                writer.writerow([
                                    str(doc_path),
                                    chunk_start,
                                    sent_start_in_doc,
                                    seen[key],
                                    tok_i,
                                    tok.lemma_,
                                    tok.text.lower(),
                                    f"{start_in_sent}:{end_in_sent}",
                                    sent_text,
                                ])
                        
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
    
    except KeyboardInterrupt:
        stopped_by_user = True
        elapsed_all = time.time() - overall_start
        logger.warning("Extraction interrupted by user (Ctrl+C)")

    # Compute checksums and save metadata
    logger.info("Computing file checksums...")
    input_checksums = {str(p): compute_file_md5(p) for p in paths if p.exists()}
    output_checksum = compute_file_md5(out_path)
    
    metadata_args = {
        "model": args.model,
        "encoding": args.encoding,
        "include_aux": args.include_aux,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "dedupe_window": args.dedupe_window,
        "heartbeat_chunks": args.heartbeat_chunks,
        "tsv": args.tsv,
    }
    
    stats = {
        "total_documents": overall_docs,
        "total_chunks": overall_chunks,
        "total_sentences": overall_sents,
        "total_verbs": overall_verbs,
        "output_rows": overall_verbs,
        "elapsed_seconds": elapsed_all,
    }
    
    status = "stopped_by_user" if stopped_by_user else "completed"
    save_run_metadata(out_path, paths, input_checksums, output_checksum, metadata_args, stats, status=status)
    logger.info(f"Wrote output: {out_path}")
    logger.info(f"Wrote metadata: {out_path.with_suffix('.json')}")


# ============================================================================
# GUI MODE
# ============================================================================

def run_gui() -> None:
    """Launch the GUI application."""
    try:
        from PySide6.QtCore import Qt, QThread, Signal, QSize
        from PySide6.QtWidgets import (
            QApplication,
            QMainWindow,
            QWidget,
            QVBoxLayout,
            QHBoxLayout,
            QGroupBox,
            QLabel,
            QLineEdit,
            QSpinBox,
            QCheckBox,
            QPushButton,
            QFileDialog,
            QTextEdit,
            QComboBox,
            QListWidget,
            QListWidgetItem,
            QProgressBar,
            QSplitter,
            QMessageBox,
        )
    except ImportError:
        print("Error: PySide6 is required for GUI mode.")
        print("Install it with: pip install PySide6")
        print("\nOr run in CLI mode by providing arguments. Use --help for usage.")
        sys.exit(1)

    class ExtractionWorker(QThread):
        """Worker thread for running verb extraction without blocking UI."""
        
        progress_update = Signal(str)
        progress_bar_update = Signal(int, str)  # (percentage, file_info_text)
        verb_count_update = Signal(int)
        sent_count_update = Signal(int)
        chunk_count_update = Signal(int)
        finished = Signal(bool, str)
        
        def __init__(
            self,
            paths: List[Path],
            output_path: Path,
            model: str,
            encoding: str,
            include_aux: bool,
            chunk_size: int,
            overlap: int,
            dedupe_window: int,
            heartbeat_chunks: int,
            use_tsv: bool,
        ):
            super().__init__()
            self.paths = paths
            self.output_path = output_path
            self.model = model
            self.encoding = encoding
            self.include_aux = include_aux
            self.chunk_size = chunk_size
            self.overlap = overlap
            self.dedupe_window = dedupe_window
            self.heartbeat_chunks = heartbeat_chunks
            self.use_tsv = use_tsv
            self._stop_requested = False
            self.total_bytes = 0
            self.bytes_processed = 0
        
        def request_stop(self):
            """Request graceful stop."""
            self._stop_requested = True
        
        def _format_bytes(self, num_bytes: int) -> str:
            """Format bytes as human-readable string (B, KB, MB, GB)."""
            for unit in ['B', 'KB', 'MB', 'GB']:
                if num_bytes < 1024.0:
                    return f"{num_bytes:.1f} {unit}"
                num_bytes /= 1024.0
            return f"{num_bytes:.1f} TB"
        
        def _process_chunk_batch(self, nlp, chunk_batch, chunk_counter, overall_chunks,
                                 file_index, file_size_bytes, seen, order, 
                                 sent_counter, verb_counter, writer, doc_path):
            """Process a batch of chunks using nlp.pipe() for efficiency."""
            # Extract texts and metadata from batch
            chunk_texts = [chunk_text for _, chunk_text in chunk_batch]
            chunk_starts = [chunk_start for chunk_start, _ in chunk_batch]
            
            # Process all chunks in batch using nlp.pipe() (uses multiple cores)
            for chunk_idx, doc in enumerate(nlp.pipe(chunk_texts)):
                chunk_start = chunk_starts[chunk_idx]
                chunk_counter += 1
                chunk_end = chunk_start + len(chunk_texts[chunk_idx])
                
                if file_size_bytes > 0:
                    pct = min(100.0, (chunk_end / file_size_bytes) * 100.0)
                else:
                    pct = 0.0
                
                self.progress_update.emit(
                    f"  Chunk {chunk_counter:6d} | chars {chunk_start:,}–{chunk_end:,} | ~{pct:6.2f}%"
                )
                
                # Update progress bar using chunk position (unaffected by overlaps)
                bytes_before_this_file = sum(
                    p.stat().st_size for p in self.paths[:file_index - 1] if p.exists()
                )
                bytes_in_this_file = chunk_end
                self.bytes_processed = bytes_before_this_file + bytes_in_this_file
                
                overall_pct = min(100, int((self.bytes_processed / self.total_bytes) * 100))
                file_info = f"File {file_index} of {len(self.paths)} | {self._format_bytes(self.bytes_processed)} / {self._format_bytes(self.total_bytes)}"
                self.progress_bar_update.emit(overall_pct, file_info)
                
                # Extract verbs from sentences
                for sent in doc.sents:
                    sent_start_in_doc = chunk_start + sent.start_char
                    key = (sent_start_in_doc, sent.text)
                    
                    if key in seen:
                        continue
                    
                    seen[key] = sent_counter
                    order.append(key)
                    sent_counter += 1
                    
                    if len(order) > self.dedupe_window:
                        drop_n = max(1, self.dedupe_window // 10)
                        for _ in range(drop_n):
                            old = order.pop(0)
                            seen.pop(old, None)
                    
                    sent_text = sent.text
                    
                    for tok_i, tok in enumerate(sent):
                        is_verb = tok.pos_ == "VERB" or (self.include_aux and tok.pos_ == "AUX")
                        if not is_verb:
                            continue
                        
                        verb_counter += 1
                        
                        start_in_sent = tok.idx - sent.start_char
                        end_in_sent = start_in_sent + len(tok.text)
                        
                        writer.writerow([
                            str(doc_path),
                            chunk_start,
                            sent_start_in_doc,
                            seen[key],
                            tok_i,
                            tok.lemma_,
                            tok.text.lower(),
                            f"{start_in_sent}:{end_in_sent}",
                            sent_text,
                        ])
                
                if self.heartbeat_chunks > 0 and (chunk_counter % self.heartbeat_chunks == 0):
                    self.progress_update.emit(
                        f"    Heartbeat: {sent_counter:,} sentences | {verb_counter:,} verbs"
                    )
                    self.sent_count_update.emit(sent_counter)
                    self.verb_count_update.emit(verb_counter)
                
                self.chunk_count_update.emit(overall_chunks + chunk_idx)
            
            return chunk_counter, sent_counter, verb_counter
        
        def run(self):
            """Run the extraction in the worker thread."""
            try:
                # Load spaCy model (with auto-download if needed)
                self.progress_update.emit(f"Loading spaCy model: {self.model}")
                nlp = load_spacy_model(self.model, logger=None)
                # Enable only components needed for verb extraction (tagger, parser, lemmatizer, etc.)
                # This is more reliable than disabling specific components
                components_to_enable = [
                    c for c in ["tok2vec", "tagger", "parser", "senter", "attribute_ruler", "lemmatizer"]
                    if c in nlp.pipe_names
                ]
                if components_to_enable:
                    nlp.select_pipes(enable=components_to_enable)
                if "parser" not in nlp.pipe_names:
                    if "sentencizer" not in nlp.pipe_names:
                        nlp.add_pipe("sentencizer")
                self.progress_update.emit(f"spaCy pipeline: {nlp.pipe_names}")
                
                # Calculate total file size for progress tracking
                self.total_bytes = 0
                for path in self.paths:
                    if path.exists():
                        try:
                            self.total_bytes += path.stat().st_size
                        except OSError:
                            pass
                
                if self.total_bytes == 0:
                    self.total_bytes = 1  # Avoid division by zero
                
                delimiter = "\t" if self.use_tsv else ","
                
                overall_verbs = 0
                overall_sents = 0
                overall_chunks = 0
                overall_docs = 0
                
                with self.output_path.open("w", encoding="utf-8", newline="") as f:
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
                    
                    for file_index, doc_path in enumerate(self.paths, 1):
                        if self._stop_requested:
                            break
                        
                        if not doc_path.exists():
                            self.progress_update.emit(f"⚠ Skipping missing path: {doc_path}")
                            continue
                        
                        overall_docs += 1
                        self.progress_update.emit(f"Starting document: {doc_path} ({file_index} of {len(self.paths)})")
                        
                        try:
                            file_size_bytes = doc_path.stat().st_size
                        except OSError:
                            file_size_bytes = 0
                        
                        seen: Dict[Tuple[int, str], int] = {}
                        order: List[Tuple[int, str]] = []
                        sent_counter = 0
                        verb_counter = 0
                        chunk_counter = 0
                        
                        # Accumulate chunks for batch processing with nlp.pipe()
                        chunk_batch = []  # List of (chunk_start, chunk_text) tuples
                        batch_size = 32  # Process chunks in batches for efficiency
                        
                        for chunk_start, chunk_text in stream_char_chunks(
                            doc_path,
                            encoding=self.encoding,
                            chunk_size=self.chunk_size,
                            overlap=self.overlap,
                            stop_check=lambda: self._stop_requested,
                        ):
                            if self._stop_requested:
                                break
                            
                            chunk_batch.append((chunk_start, chunk_text))
                            
                            # Process batch when it reaches batch_size or at end of file
                            if len(chunk_batch) >= batch_size:
                                chunk_counter, sent_counter, verb_counter = self._process_chunk_batch(
                                    nlp, chunk_batch, chunk_counter, overall_chunks,
                                    file_index, file_size_bytes, 
                                    seen, order, sent_counter, verb_counter,
                                    writer, doc_path
                                )
                                chunk_batch = []
                        
                        # Process remaining chunks in final batch
                        if chunk_batch and not self._stop_requested:
                            chunk_counter, sent_counter, verb_counter = self._process_chunk_batch(
                                nlp, chunk_batch, chunk_counter, overall_chunks,
                                file_index, file_size_bytes,
                                seen, order, sent_counter, verb_counter,
                                writer, doc_path
                            )
                        
                        overall_sents += sent_counter
                        overall_verbs += verb_counter
                        
                        self.progress_update.emit(
                            f"Finished document: {doc_path} | "
                            f"{chunk_counter:,} chunks | {sent_counter:,} sentences | {verb_counter:,} verbs"
                        )
                
                self.progress_update.emit(
                    f"✓ Run complete | {overall_chunks:,} chunks | {overall_sents:,} sentences | {overall_verbs:,} verbs"
                )
                
                # Determine if run was stopped or completed
                status = "stopped_by_user" if self._stop_requested else "completed"
                
                # Save metadata
                input_checksums = {str(p): compute_file_md5(p) for p in self.paths if p.exists()}
                output_checksum = compute_file_md5(self.output_path)
                
                metadata_args = {
                    "model": self.model,
                    "encoding": self.encoding,
                    "include_aux": self.include_aux,
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap,
                    "dedupe_window": self.dedupe_window,
                    "heartbeat_chunks": self.heartbeat_chunks,
                    "tsv": self.use_tsv,
                }
                
                stats = {
                    "total_chunks": overall_chunks,
                    "total_sentences": overall_sents,
                    "total_verbs": overall_verbs,
                    "output_rows": overall_verbs,
                }
                
                save_run_metadata(self.output_path, self.paths, input_checksums, output_checksum, metadata_args, stats, status=status)
                self.progress_update.emit(f"✓ Saved metadata: {self.output_path.with_suffix('.json')}")
                
                self.finished.emit(True, f"Extraction complete. Output: {self.output_path}")
                
            except Exception as e:
                self.progress_update.emit(f"✗ Error: {str(e)}")
                self.finished.emit(False, f"Error: {str(e)}")

    class SpaCyVerbExtractorGUI(QMainWindow):
        """Main GUI window for SpaCy verb extractor."""
        
        def __init__(self):
            super().__init__()
            self.setWindowTitle("SpaCy Verb Extractor")
            self.setMinimumSize(QSize(1000, 700))
            
            self.input_files: List[Path] = []
            self.worker: Optional[ExtractionWorker] = None
            
            self.init_ui()
        
        def init_ui(self):
            """Initialize the UI components."""
            central = QWidget()
            self.setCentralWidget(central)
            
            layout = QVBoxLayout(central)
            
            # Input files section
            files_group = QGroupBox("Input Files")
            files_layout = QVBoxLayout()
            
            files_button_layout = QHBoxLayout()
            self.add_files_btn = QPushButton("Add Files")
            self.add_files_btn.clicked.connect(self.add_input_files)
            self.add_paths_file_btn = QPushButton("Add Paths File")
            self.add_paths_file_btn.clicked.connect(self.add_paths_file)
            self.clear_files_btn = QPushButton("Clear List")
            self.clear_files_btn.clicked.connect(self.clear_input_files)
            files_button_layout.addWidget(self.add_files_btn)
            files_button_layout.addWidget(self.add_paths_file_btn)
            files_button_layout.addWidget(self.clear_files_btn)
            files_button_layout.addStretch()
            
            files_layout.addLayout(files_button_layout)
            
            self.files_list = QListWidget()
            files_layout.addWidget(self.files_list)
            
            files_group.setLayout(files_layout)
            layout.addWidget(files_group)
            
            # Settings section
            settings_group = QGroupBox("Settings")
            settings_layout = QVBoxLayout()
            
            # Load metadata button
            load_metadata_layout = QHBoxLayout()
            self.load_metadata_btn = QPushButton("Load Settings from JSON")
            self.load_metadata_btn.clicked.connect(self.load_metadata_dialog)
            load_metadata_layout.addWidget(self.load_metadata_btn)
            load_metadata_layout.addStretch()
            settings_layout.addLayout(load_metadata_layout)
            
            # Output path
            output_layout = QHBoxLayout()
            output_label = QLabel("Output File:")
            self.output_input = QLineEdit("verbs.csv")
            self.output_browse = QPushButton("Browse...")
            self.output_browse.clicked.connect(self.browse_output)
            output_layout.addWidget(output_label)
            output_layout.addWidget(self.output_input)
            output_layout.addWidget(self.output_browse)
            settings_layout.addLayout(output_layout)
            
            # Model and encoding
            model_encoding_layout = QHBoxLayout()
            model_layout = QVBoxLayout()
            model_label = QLabel("Model:")
            self.model_combo = QComboBox()
            self.model_combo.addItems([
                "en_core_web_sm",
                "en_core_web_md",
                "en_core_web_lg",
                "en_core_web_trf",
            ])
            model_layout.addWidget(model_label)
            model_layout.addWidget(self.model_combo)
            
            encoding_layout = QVBoxLayout()
            encoding_label = QLabel("Encoding:")
            self.encoding_combo = QComboBox()
            self.encoding_combo.addItems(["utf-8", "utf-8-sig", "latin-1", "cp1252"])
            encoding_layout.addWidget(encoding_label)
            encoding_layout.addWidget(self.encoding_combo)
            
            model_encoding_layout.addLayout(model_layout)
            model_encoding_layout.addLayout(encoding_layout)
            settings_layout.addLayout(model_encoding_layout)
            
            # Chunk settings
            chunk_layout = QHBoxLayout()
            
            chunk_size_layout = QVBoxLayout()
            chunk_size_label = QLabel("Chunk Size (chars):")
            self.chunk_size_spin = QSpinBox()
            self.chunk_size_spin.setMinimum(10000)
            self.chunk_size_spin.setMaximum(10000000)
            self.chunk_size_spin.setValue(2000000)
            self.chunk_size_spin.setSingleStep(100000)
            chunk_size_layout.addWidget(chunk_size_label)
            chunk_size_layout.addWidget(self.chunk_size_spin)
            
            overlap_layout = QVBoxLayout()
            overlap_label = QLabel("Overlap (chars):")
            self.overlap_spin = QSpinBox()
            self.overlap_spin.setMinimum(0)
            self.overlap_spin.setMaximum(100000)
            self.overlap_spin.setValue(5000)
            overlap_layout.addWidget(overlap_label)
            overlap_layout.addWidget(self.overlap_spin)
            
            chunk_layout.addLayout(chunk_size_layout)
            chunk_layout.addLayout(overlap_layout)
            chunk_layout.addStretch()
            settings_layout.addLayout(chunk_layout)
            
            # Dedupe and heartbeat
            dedupe_layout = QHBoxLayout()
            
            dedupe_window_layout = QVBoxLayout()
            dedupe_label = QLabel("Dedupe Window (sents):")
            self.dedupe_spin = QSpinBox()
            self.dedupe_spin.setMinimum(1000)
            self.dedupe_spin.setMaximum(500000)
            self.dedupe_spin.setValue(50000)
            self.dedupe_spin.setSingleStep(10000)
            dedupe_window_layout.addWidget(dedupe_label)
            dedupe_window_layout.addWidget(self.dedupe_spin)
            
            heartbeat_layout = QVBoxLayout()
            heartbeat_label = QLabel("Heartbeat Interval (chunks):")
            self.heartbeat_spin = QSpinBox()
            self.heartbeat_spin.setMinimum(1)
            self.heartbeat_spin.setMaximum(100)
            self.heartbeat_spin.setValue(10)
            heartbeat_layout.addWidget(heartbeat_label)
            heartbeat_layout.addWidget(self.heartbeat_spin)
            
            dedupe_layout.addLayout(dedupe_window_layout)
            dedupe_layout.addLayout(heartbeat_layout)
            dedupe_layout.addStretch()
            settings_layout.addLayout(dedupe_layout)
            
            # Checkboxes
            checkbox_layout = QHBoxLayout()
            self.include_aux_check = QCheckBox("Include AUX tokens")
            self.tsv_check = QCheckBox("Output TSV (default: CSV)")
            checkbox_layout.addWidget(self.include_aux_check)
            checkbox_layout.addWidget(self.tsv_check)
            checkbox_layout.addStretch()
            settings_layout.addLayout(checkbox_layout)
            
            settings_group.setLayout(settings_layout)
            layout.addWidget(settings_group)
            
            # Progress section
            progress_group = QGroupBox("Progress")
            progress_layout = QVBoxLayout()
            
            self.progress_bar = QProgressBar()
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(100)
            self.progress_bar.setValue(0)
            progress_layout.addWidget(self.progress_bar)
            
            self.progress_info_label = QLabel("Ready to start")
            self.progress_info_label.setStyleSheet("color: #666; font-size: 11px;")
            progress_layout.addWidget(self.progress_info_label)
            
            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            self.log_text.setMaximumHeight(200)
            progress_layout.addWidget(QLabel("Log Output:"))
            progress_layout.addWidget(self.log_text)
            
            progress_group.setLayout(progress_layout)
            layout.addWidget(progress_group)
            
            # Control buttons
            button_layout = QHBoxLayout()
            self.start_btn = QPushButton("Start Extraction")
            self.start_btn.clicked.connect(self.start_extraction)
            self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
            self.stop_btn = QPushButton("Stop")
            self.stop_btn.clicked.connect(self.stop_extraction)
            self.stop_btn.setEnabled(False)
            self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 5px;")
            button_layout.addStretch()
            button_layout.addWidget(self.start_btn)
            button_layout.addWidget(self.stop_btn)
            button_layout.addStretch()
            layout.addLayout(button_layout)
            
            central.setLayout(layout)
        
        def add_input_files(self):
            """Open file dialog to add input files."""
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Text Files",
                "",
                "Text Files (*.txt);;All Files (*)"
            )
            for f in files:
                path = Path(f)
                if path not in self.input_files:
                    self.input_files.append(path)
                    self.files_list.addItem(str(path))
        
        def add_paths_file(self):
            """Open file dialog to add a paths file."""
            file, _ = QFileDialog.getOpenFileName(
                self,
                "Select Paths File",
                "",
                "Text Files (*.txt);;All Files (*)"
            )
            if file:
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            path = Path(line)
                            if path not in self.input_files:
                                self.input_files.append(path)
                                self.files_list.addItem(str(path))
                    self.log("Added paths from file: " + file)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to read paths file: {str(e)}")
        
        def clear_input_files(self):
            """Clear the input files list."""
            self.input_files.clear()
            self.files_list.clear()
        
        def browse_output(self):
            """Open file dialog to select output file."""
            file, _ = QFileDialog.getSaveFileName(
                self,
                "Select Output File",
                "verbs.csv",
                "CSV Files (*.csv);;TSV Files (*.tsv);;All Files (*)"
            )
            if file:
                self.output_input.setText(file)
        
        def load_metadata_dialog(self):
            """Open file dialog to load metadata JSON."""
            file, _ = QFileDialog.getOpenFileName(
                self,
                "Select Metadata JSON File",
                "",
                "JSON Files (*.json);;All Files (*)"
            )
            if file:
                self.load_metadata_from_file(Path(file))
        
        def load_metadata_from_file(self, json_path: Path):
            """Load settings from a metadata JSON file and populate GUI."""
            try:
                metadata = load_run_metadata(json_path)
                
                # Get input files from metadata
                input_files_from_json = metadata.get("input_files", [])
                
                # Verify input file checksums
                input_checksums = metadata.get("input_checksums", {})
                missing_files = []
                checksum_mismatches = []
                valid_files = []
                
                for file_path_str, expected_checksum in input_checksums.items():
                    file_path = Path(file_path_str)
                    if not file_path.exists():
                        missing_files.append(file_path_str)
                    else:
                        actual_checksum = compute_file_md5(file_path)
                        if actual_checksum != expected_checksum:
                            checksum_mismatches.append(file_path_str)
                        else:
                            valid_files.append(file_path)
                
                # Show warnings if issues found
                warnings = []
                if missing_files:
                    warnings.append(f"⚠ Missing input files:\n" + "\n".join(missing_files))
                if checksum_mismatches:
                    warnings.append(f"⚠ Input files have changed (checksum mismatch):\n" + "\n".join(checksum_mismatches))
                
                if warnings:
                    msg = "\n\n".join(warnings) + "\n\nContinue loading settings anyway?"
                    reply = QMessageBox.warning(self, "Input File Issues", msg, QMessageBox.Yes | QMessageBox.No)
                    if reply == QMessageBox.No:
                        return
                
                # Load input files into GUI
                self.input_files.clear()
                self.files_list.clear()
                
                for file_path_str in input_files_from_json:
                    file_path = Path(file_path_str)
                    if file_path not in self.input_files:
                        self.input_files.append(file_path)
                        self.files_list.addItem(str(file_path))
                
                # Load settings
                settings = metadata.get("settings", {})
                
                self.output_input.setText(metadata.get("output_file", "verbs.csv"))
                self.model_combo.setCurrentText(settings.get("model", "en_core_web_sm"))
                self.encoding_combo.setCurrentText(settings.get("encoding", "utf-8"))
                self.chunk_size_spin.setValue(settings.get("chunk_size", 2_000_000))
                self.overlap_spin.setValue(settings.get("overlap", 5_000))
                self.dedupe_spin.setValue(settings.get("dedupe_window", 50_000))
                self.heartbeat_spin.setValue(settings.get("heartbeat_chunks", 10))
                self.include_aux_check.setChecked(settings.get("include_aux", False))
                self.tsv_check.setChecked(settings.get("output_format", "csv") == "tsv")
                
                self.log(f"✓ Loaded metadata from: {json_path}")
                if warnings:
                    for w in warnings:
                        self.log(w)
                else:
                    self.log("✓ All input files verified (checksum OK)")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load metadata: {str(e)}")
        
        def log(self, message: str):
            """Append a message to the log output."""
            self.log_text.append(message)
            # Auto-scroll to bottom
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )
        
        def update_progress_bar(self, percentage: int, file_info: str):
            """Update the progress bar and info label."""
            self.progress_bar.setValue(percentage)
            self.progress_info_label.setText(file_info)
        
        def start_extraction(self):
            """Start the extraction process."""
            if not self.input_files:
                QMessageBox.warning(self, "No Input Files", "Please add at least one input file.")
                return
            
            output_path = Path(self.output_input.text()).resolve()
            
            # Validate output path
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot create output directory: {str(e)}")
                return
            
            # Check if any input files match metadata (if metadata was loaded)
            # This warning happens before clearing the log, so user sees it
            metadata_path = output_path.with_suffix(".json")
            if metadata_path.exists():
                try:
                    metadata = load_run_metadata(metadata_path)
                    issues = verify_input_file_checksums(self.input_files, metadata)
                    
                    if issues["missing_files"] or issues["checksum_mismatches"]:
                        warnings = []
                        if issues["missing_files"]:
                            warnings.append(f"⚠ Missing from metadata:\n" + "\n".join(issues["missing_files"]))
                        if issues["checksum_mismatches"]:
                            warnings.append(f"⚠ Changed since last run (checksum mismatch):\n" + "\n".join(issues["checksum_mismatches"]))
                        
                        msg = "\n\n".join(warnings) + "\n\nContinue with extraction?"
                        reply = QMessageBox.warning(self, "Input File Checksum Mismatch", msg, QMessageBox.Yes | QMessageBox.No)
                        if reply == QMessageBox.No:
                            return
                except Exception as e:
                    # Metadata issues shouldn't block extraction
                    pass
            
            # Clear log and reset progress bar
            self.log_text.clear()
            self.progress_bar.setValue(0)
            self.progress_info_label.setText("Processing...")
            self.log("=" * 60)
            self.log("Starting Verb Extraction")
            self.log("=" * 60)
            self.log(f"Input files: {len(self.input_files)}")
            self.log(f"Output: {output_path}")
            self.log("")
            
            # Disable controls
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.add_files_btn.setEnabled(False)
            self.add_paths_file_btn.setEnabled(False)
            
            # Create and start worker
            self.worker = ExtractionWorker(
                paths=self.input_files,
                output_path=output_path,
                model=self.model_combo.currentText(),
                encoding=self.encoding_combo.currentText(),
                include_aux=self.include_aux_check.isChecked(),
                chunk_size=self.chunk_size_spin.value(),
                overlap=self.overlap_spin.value(),
                dedupe_window=self.dedupe_spin.value(),
                heartbeat_chunks=self.heartbeat_spin.value(),
                use_tsv=self.tsv_check.isChecked(),
            )
            
            self.worker.progress_update.connect(self.log)
            self.worker.progress_bar_update.connect(self.update_progress_bar)
            self.worker.verb_count_update.connect(lambda v: self.log(f"Verbs: {v:,}"))
            self.worker.sent_count_update.connect(lambda s: self.log(f"Sentences: {s:,}"))
            self.worker.chunk_count_update.connect(lambda c: self.log(f"Chunks: {c:,}"))
            self.worker.finished.connect(self.on_extraction_finished)
            
            self.worker.start()
        
        def stop_extraction(self):
            """Stop the running extraction."""
            if self.worker:
                self.log("Stopping extraction...")
                self.worker.request_stop()
                self.worker.wait()
                self.on_extraction_finished(False, "Extraction stopped by user")
        
        def on_extraction_finished(self, success: bool, message: str):
            """Handle extraction completion."""
            # Re-enable controls
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.add_files_btn.setEnabled(True)
            self.add_paths_file_btn.setEnabled(True)
            
            # Update progress display
            if success:
                self.progress_bar.setValue(100)
                self.progress_info_label.setText("Complete")
            else:
                self.progress_info_label.setText("Error")
            
            self.log("")
            self.log("=" * 60)
            self.log(message)
            self.log("=" * 60)
            
            if success:
                QMessageBox.information(self, "Success", message)
            else:
                QMessageBox.critical(self, "Error", message)

    app = QApplication(sys.argv)
    window = SpaCyVerbExtractorGUI()
    window.show()
    sys.exit(app.exec())


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point - choose GUI or CLI mode based on arguments."""
    # If no arguments (or just the script name), run GUI
    # Otherwise, run CLI
    if len(sys.argv) == 1:
        print("Launching GUI mode...")
        print("(Use command-line arguments to run in CLI mode. Use --help for usage.)")
        run_gui()
    else:
        run_cli()


if __name__ == "__main__":
    main()
