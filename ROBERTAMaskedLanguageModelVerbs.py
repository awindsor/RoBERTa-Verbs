#!/usr/bin/env python3
"""
Run RoBERTa masked-LM inference on a "verbs file" CSV with CLI + GUI + metadata support.

Supports both CLI and GUI modes:
  - Run without arguments: launches GUI (requires PySide6)
  - Run with arguments: uses CLI mode

For each row:
  - read sentence + span_in_sentence_char (format "start:end")
  - replace that character span with RoBERTa's mask token (<mask>)
  - run masked language model inference
  - write the original row plus 2*top_k new columns:
      token_1, prob_1, token_2, prob_2, ..., token_k, prob_k

Output includes JSON metadata with:
  - All inference settings
  - MD5 checksums of input/output
  - Inference statistics

Streaming-friendly:
  - reads input CSV line-by-line
  - writes output CSV line-by-line
  - holds only a small batch in memory

Requirements:
  pip install transformers torch
  pip install PySide6  # for GUI mode only

CLI Examples:
  python RoBERTaMaskedLanguageModelVerbs.py verbs.csv output.csv --model roberta-base --batch-size 16 --top-k 10
  python RoBERTaMaskedLanguageModelVerbs.py verbs.csv output.csv --device cuda --top-k 20
  python RoBERTaMaskedLanguageModelVerbs.py --load-metadata output.json verbs.csv output2.csv

GUI Mode:
  python RoBERTaMaskedLanguageModelVerbs.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


# ============================================================================
# METADATA AND UTILITY FUNCTIONS
# ============================================================================

def compute_file_md5(path: Path) -> str:
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def save_mlm_metadata(
    output_path: Path,
    input_path: Path,
    input_checksum: str,
    output_checksum: str,
    settings: Dict,
    stats: Dict,
    source_metadata: Optional[Dict] = None,
) -> None:
    """Save MLM inference metadata to JSON file alongside output.
    
    Args:
        source_metadata: Metadata from previous tool (e.g., SpaCyVerbExtractor) for chaining
    """
    json_path = output_path.with_suffix(".json")
    
    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tool": "RoBERTaMaskedLanguageModelVerbs",
        "input_file": str(input_path),
        "input_checksum": input_checksum,
        "output_file": str(output_path),
        "output_checksum": output_checksum,
        "settings": settings,
        "statistics": stats,
    }
    
    # Include source metadata for pipeline traceability
    if source_metadata:
        metadata["source_metadata"] = source_metadata
    
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_mlm_metadata(json_path: Path) -> Dict:
    """Load MLM metadata from JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify_input_checksum(input_path: Path, metadata: Dict) -> Tuple[bool, str]:
    """
    Verify input file checksum against metadata.
    
    Returns (matches: bool, message: str)
    """
    expected_checksum = metadata.get("input_checksum")
    if not expected_checksum:
        return True, "No checksum in metadata"
    
    if not input_path.exists():
        return False, f"Input file not found: {input_path}"
    
    actual_checksum = compute_file_md5(input_path)
    if actual_checksum != expected_checksum:
        return False, f"Input file has changed (checksum mismatch): {input_path}"
    
    return True, "Input file verified (checksum OK)"


# ============================================================================
# CORE MLM FUNCTIONS (shared by CLI and GUI)
# ============================================================================

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


# ============================================================================
# CLI MODE
# ============================================================================

def run_cli() -> None:
    """Run in CLI mode."""
    ap = argparse.ArgumentParser(
        description="Run RoBERTa masked-LM inference on verb CSV"
    )
    ap.add_argument("input_csv", nargs="?", help="Verbs file CSV (from your extractor)")
    ap.add_argument("output_csv", nargs="?", help="Output CSV with MLM predictions appended")
    ap.add_argument("--model", default="roberta-base", help="HF model name (default: roberta-base)")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    ap.add_argument("--top-k", type=int, default=10, help="Top-k predictions to write (default: 10)")
    ap.add_argument("--device", default=None, help='Device: "cpu", "cuda", "mps", or leave blank for auto')
    ap.add_argument("--log-every", type=int, default=1000, help="Log progress every N rows (default: 1000)")
    ap.add_argument("--log-level", default="INFO", help='Logging level: DEBUG, INFO, WARNING (default: INFO)')
    ap.add_argument("--debug-limit", type=int, default=0,
                    help="If >0, stop after this many rows seen (processed+skipped). Handy for debugging (e.g., 100).")
    ap.add_argument("--encoding", default="utf-8", help="CSV encoding (default: utf-8)")
    ap.add_argument("--load-metadata", help="Load settings from metadata JSON (CLI args override)")
    args = ap.parse_args()

    # Load metadata if provided
    metadata = None
    source_metadata = None  # Store the source metadata for chaining
    
    if args.load_metadata:
        metadata_path = Path(args.load_metadata)
        if not metadata_path.exists():
            raise SystemExit(f"Metadata file not found: {metadata_path}")
        
        metadata = load_mlm_metadata(metadata_path)
        tool_name = metadata.get("tool", "unknown")
        
        # Check if this is a SpaCyVerbExtractor or FilterSpaCyVerbs JSON
        if tool_name == "SpaCyVerbExtractor":
            # SpaCyVerbExtractor output - use its output as our input
            source_metadata = metadata  # Save for chaining
            args.input_csv = args.input_csv or metadata.get("output_file")
        elif tool_name == "FilterSpaCyVerbs":
            # FilterSpaCyVerbs output - use its output as our input
            source_metadata = metadata  # Save for chaining
            args.input_csv = args.input_csv or metadata.get("output_file")
        elif tool_name == "RoBERTaMaskedLanguageModelVerbs":
            # This is existing MLM metadata - use its input
            source_metadata = metadata.get("source_metadata")  # Preserve chain
            args.input_csv = args.input_csv or metadata.get("input_file")
        else:
            # Generic metadata - use input_file if available
            if "input_file" in metadata:
                args.input_csv = args.input_csv or metadata.get("input_file")
        
        # Use loaded settings as defaults, CLI args override
        if not args.model or args.model == "roberta-base":
            args.model = metadata.get("settings", {}).get("model", "roberta-base")
        if args.batch_size == 16:
            args.batch_size = metadata.get("settings", {}).get("batch_size", 16)
        if args.top_k == 10:
            args.top_k = metadata.get("settings", {}).get("top_k", 10)
        if not args.device:
            args.device = metadata.get("settings", {}).get("device")
        
        # If input not provided on CLI, use from metadata
        if not args.input_csv:
            args.input_csv = metadata.get("input_file")
    
    # Validate required arguments
    if not args.input_csv or not args.output_csv:
        ap.print_help()
        raise SystemExit("Error: input_csv and output_csv are required (or use --load-metadata)")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("mlm")
    
    # Log metadata source if loaded
    if metadata:
        tool_name = metadata.get("tool", "unknown")
        logger.info(f"This metadata is from '{tool_name}'")

    if args.top_k <= 0:
        raise SystemExit("--top-k must be a positive integer.")

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    
    # Verify checksum if metadata was loaded
    if metadata:
        matches, message = verify_input_checksum(input_path, metadata)
        if not matches:
            print(f"⚠ {message}")

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

    start_time = time.time()

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

    elapsed_time = time.time() - start_time
    
    # Compute checksums and save metadata
    input_checksum = compute_file_md5(input_path)
    output_checksum = compute_file_md5(output_path)
    
    mlm_settings = {
        "model": args.model,
        "batch_size": args.batch_size,
        "top_k": args.top_k,
        "device": str(dev),
        "encoding": args.encoding,
        "debug_limit": args.debug_limit if args.debug_limit else None,
    }
    
    stats = {
        "rows_seen": seen,
        "rows_written": processed,
        "rows_skipped": skipped,
        "elapsed_seconds": round(elapsed_time, 2),
    }
    
    save_mlm_metadata(output_path, input_path, input_checksum, output_checksum, mlm_settings, stats, source_metadata)

    logger.info(f"Done. Written={processed:,} skipped={skipped:,} time={elapsed_time:.1f}s output={args.output_csv}")
    logger.info(f"✓ Saved metadata to {output_path.with_suffix('.json')}")


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
            QComboBox,
            QPushButton,
            QFileDialog,
            QTextEdit,
            QMessageBox,
            QCheckBox,
        )
    except ImportError:
        print("Error: PySide6 is required for GUI mode.")
        print("Install it with: pip install PySide6")
        print("\nOr run in CLI mode by providing arguments. Use --help for usage.")
        sys.exit(1)

    class MLMWorker(QThread):
        """Worker thread for MLM inference without blocking UI."""
        
        progress_update = Signal(str)
        finished = Signal(bool, str)
        
        def __init__(
            self,
            input_path: Path,
            output_path: Path,
            model_name: str,
            batch_size: int,
            top_k: int,
            device_name: str,
            debug_limit: int,
        ):
            super().__init__()
            self.input_path = input_path
            self.output_path = output_path
            self.model_name = model_name
            self.batch_size = batch_size
            self.top_k = top_k
            self.device_name = device_name
            self.debug_limit = debug_limit
            self._stop_requested = False
        
        def request_stop(self):
            """Request graceful stop."""
            self._stop_requested = True
        
        def run(self):
            """Run the MLM inference in the worker thread."""
            try:
                # Device selection
                if self.device_name and self.device_name != "auto":
                    dev = torch.device(self.device_name)
                else:
                    if torch.cuda.is_available():
                        dev = torch.device("cuda")
                    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                        dev = torch.device("mps")
                    else:
                        dev = torch.device("cpu")
                
                self.progress_update.emit(f"Loading model: {self.model_name}")
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForMaskedLM.from_pretrained(self.model_name)
                model.to(dev)
                model.eval()
                
                if tokenizer.mask_token is None:
                    raise ValueError("Tokenizer has no mask_token")
                
                mask_token = tokenizer.mask_token
                self.progress_update.emit(f"Device: {dev}, Mask token: {mask_token}")
                self.progress_update.emit(f"Processing with batch_size={self.batch_size}, top_k={self.top_k}")
                
                pred_cols: List[str] = []
                for i in range(1, self.top_k + 1):
                    pred_cols.extend([f"token_{i}", f"prob_{i}"])
                
                needed_cols = {"sentence", "span_in_sentence_char"}
                processed = 0
                skipped = 0
                seen = 0
                pending: List[PendingRow] = []
                start_time = time.time()
                
                with self.input_path.open(newline="", encoding="utf-8") as fin, \
                     self.output_path.open("w", newline="", encoding="utf-8") as fout:
                    
                    reader = csv.DictReader(fin)
                    if reader.fieldnames is None:
                        raise ValueError("Input CSV has no header")
                    
                    missing = needed_cols - set(reader.fieldnames)
                    if missing:
                        raise ValueError(f"Missing required columns: {sorted(missing)}")
                    
                    out_fieldnames = list(reader.fieldnames) + pred_cols
                    writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
                    writer.writeheader()
                    
                    def flush_batch(batch: List[PendingRow]) -> None:
                        nonlocal processed
                        texts = [pr.masked_text for pr in batch]
                        preds = topk_for_batch(model, tokenizer, texts, self.top_k, dev)
                        
                        for pr, pr_preds in zip(batch, preds):
                            out_row = dict(pr.row)
                            for j, (tok, prob) in enumerate(pr_preds, start=1):
                                out_row[f"token_{j}"] = tok
                                out_row[f"prob_{j}"] = f"{prob:.10g}"
                            writer.writerow(out_row)
                            processed += 1
                    
                    for row_idx, row in enumerate(reader, start=1):
                        if self._stop_requested:
                            raise ValueError("Stop requested by user")
                        
                        seen += 1
                        
                        if self.debug_limit and (processed + skipped) >= self.debug_limit:
                            self.progress_update.emit(f"Debug limit reached ({self.debug_limit})")
                            break
                        
                        try:
                            raw_sent = row["sentence"]
                            span = row["span_in_sentence_char"]
                            masked_raw = mask_sentence(raw_sent, span, mask_token)
                            masked = " " + masked_raw.strip()
                            pending.append(PendingRow(row=row, masked_text=masked))
                        except Exception as e:
                            skipped += 1
                            if skipped <= 10:
                                self.progress_update.emit(f"⚠ Skipping row {row_idx}: {e}")
                            continue
                        
                        if len(pending) >= self.batch_size:
                            flush_batch(pending)
                            pending.clear()
                            if seen % 1000 == 0:
                                self.progress_update.emit(f"Processed {seen:,} rows ({processed:,} written, {skipped:,} skipped)")
                    
                    if pending and (not self.debug_limit or (processed + skipped) < self.debug_limit):
                        flush_batch(pending)
                        pending.clear()
                
                elapsed_time = time.time() - start_time
                
                # Save metadata
                self.progress_update.emit("Writing metadata...")
                input_checksum = compute_file_md5(self.input_path)
                output_checksum = compute_file_md5(self.output_path)
                
                mlm_settings = {
                    "model": self.model_name,
                    "batch_size": self.batch_size,
                    "top_k": self.top_k,
                    "device": str(dev),
                    "debug_limit": self.debug_limit if self.debug_limit else None,
                }
                
                stats = {
                    "rows_seen": seen,
                    "rows_written": processed,
                    "rows_skipped": skipped,
                    "elapsed_seconds": round(elapsed_time, 2),
                }
                
                save_mlm_metadata(self.output_path, self.input_path, input_checksum, output_checksum, mlm_settings, stats, None)
                
                self.progress_update.emit(f"✓ Written {processed:,} rows in {elapsed_time:.1f}s")
                self.progress_update.emit(f"✓ Saved metadata: {self.output_path.with_suffix('.json')}")
                self.finished.emit(True, f"MLM inference complete. Output: {self.output_path}")
                
            except Exception as e:
                self.progress_update.emit(f"✗ Error: {str(e)}")
                self.finished.emit(False, f"Error: {str(e)}")

    class MLMInferenceGUI(QMainWindow):
        """Main GUI window for MLM inference tool."""
        
        def __init__(self):
            super().__init__()
            self.setWindowTitle("RoBERTa MLM Inference")
            self.setMinimumSize(QSize(950, 700))
            
            self.worker: Optional[MLMWorker] = None
            
            self.init_ui()
        
        def init_ui(self):
            """Initialize the UI components."""
            central = QWidget()
            self.setCentralWidget(central)
            
            layout = QVBoxLayout(central)
            
            # File section
            files_group = QGroupBox("Files")
            files_layout = QVBoxLayout()
            
            # Input file
            input_layout = QHBoxLayout()
            input_label = QLabel("Input CSV:")
            self.input_text = QLineEdit()
            self.input_browse = QPushButton("Browse...")
            self.input_browse.clicked.connect(self.browse_input)
            input_layout.addWidget(input_label)
            input_layout.addWidget(self.input_text)
            input_layout.addWidget(self.input_browse)
            files_layout.addLayout(input_layout)
            
            # Output file
            output_layout = QHBoxLayout()
            output_label = QLabel("Output CSV:")
            self.output_text = QLineEdit("verbs_with_mlm.csv")
            self.output_browse = QPushButton("Browse...")
            self.output_browse.clicked.connect(self.browse_output)
            output_layout.addWidget(output_label)
            output_layout.addWidget(self.output_text)
            output_layout.addWidget(self.output_browse)
            files_layout.addLayout(output_layout)
            
            files_group.setLayout(files_layout)
            layout.addWidget(files_group)
            
            # Settings section
            settings_group = QGroupBox("Settings")
            settings_layout = QVBoxLayout()
            
            # Load metadata button
            metadata_layout = QHBoxLayout()
            self.load_metadata_btn = QPushButton("Load Settings from JSON")
            self.load_metadata_btn.clicked.connect(self.load_metadata_dialog)
            metadata_layout.addWidget(self.load_metadata_btn)
            metadata_layout.addStretch()
            settings_layout.addLayout(metadata_layout)
            
            # Model selection
            model_layout = QHBoxLayout()
            model_label = QLabel("Model:")
            self.model_combo = QComboBox()
            self.model_combo.setEditable(True)
            self.model_combo.addItems(["roberta-base", "roberta-large", "distilroberta-base"])
            model_layout.addWidget(model_label)
            model_layout.addWidget(self.model_combo)
            model_layout.addStretch()
            settings_layout.addLayout(model_layout)
            
            # Batch size and top-k
            params_layout = QHBoxLayout()
            
            batch_layout = QVBoxLayout()
            batch_label = QLabel("Batch Size:")
            self.batch_spin = QSpinBox()
            self.batch_spin.setMinimum(1)
            self.batch_spin.setMaximum(128)
            self.batch_spin.setValue(16)
            batch_layout.addWidget(batch_label)
            batch_layout.addWidget(self.batch_spin)
            
            topk_layout = QVBoxLayout()
            topk_label = QLabel("Top-K:")
            self.topk_spin = QSpinBox()
            self.topk_spin.setMinimum(1)
            self.topk_spin.setMaximum(100)
            self.topk_spin.setValue(10)
            topk_layout.addWidget(topk_label)
            topk_layout.addWidget(self.topk_spin)
            
            params_layout.addLayout(batch_layout)
            params_layout.addLayout(topk_layout)
            params_layout.addStretch()
            settings_layout.addLayout(params_layout)
            
            # Device selection
            device_layout = QHBoxLayout()
            device_label = QLabel("Device:")
            self.device_combo = QComboBox()
            self.device_combo.addItems(["auto", "cpu", "cuda", "mps"])
            device_layout.addWidget(device_label)
            device_layout.addWidget(self.device_combo)
            device_layout.addStretch()
            settings_layout.addLayout(device_layout)
            
            # Debug limit
            debug_layout = QHBoxLayout()
            self.debug_check = QCheckBox("Debug Mode (limit rows)")
            self.debug_spin = QSpinBox()
            self.debug_spin.setMinimum(0)
            self.debug_spin.setMaximum(100000)
            self.debug_spin.setValue(0)
            self.debug_spin.setEnabled(False)
            self.debug_check.toggled.connect(self.debug_spin.setEnabled)
            debug_layout.addWidget(self.debug_check)
            debug_layout.addWidget(self.debug_spin)
            debug_layout.addStretch()
            settings_layout.addLayout(debug_layout)
            
            settings_group.setLayout(settings_layout)
            layout.addWidget(settings_group)
            
            # Progress section
            progress_group = QGroupBox("Progress")
            progress_layout = QVBoxLayout()
            
            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            self.log_text.setMaximumHeight(200)
            progress_layout.addWidget(self.log_text)
            
            progress_group.setLayout(progress_layout)
            layout.addWidget(progress_group)
            
            # Control buttons
            button_layout = QHBoxLayout()
            self.start_btn = QPushButton("Start MLM Inference")
            self.start_btn.clicked.connect(self.start_inference)
            self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
            self.stop_btn = QPushButton("Stop")
            self.stop_btn.clicked.connect(self.stop_inference)
            self.stop_btn.setEnabled(False)
            self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 5px;")
            button_layout.addStretch()
            button_layout.addWidget(self.start_btn)
            button_layout.addWidget(self.stop_btn)
            button_layout.addStretch()
            layout.addLayout(button_layout)
            
            central.setLayout(layout)
        
        def browse_input(self):
            """Open file dialog for input file."""
            file, _ = QFileDialog.getOpenFileName(
                self,
                "Select Input CSV",
                "",
                "CSV Files (*.csv);;All Files (*)"
            )
            if file:
                self.input_text.setText(file)
        
        def browse_output(self):
            """Open file dialog for output file."""
            file, _ = QFileDialog.getSaveFileName(
                self,
                "Select Output CSV",
                "verbs_with_mlm.csv",
                "CSV Files (*.csv);;All Files (*)"
            )
            if file:
                self.output_text.setText(file)
        
        def load_metadata_dialog(self):
            """Open file dialog to load metadata JSON."""
            file, _ = QFileDialog.getOpenFileName(
                self,
                "Select Metadata JSON",
                "",
                "JSON Files (*.json);;All Files (*)"
            )
            if file:
                self.load_metadata_from_file(Path(file))
        
        def load_metadata_from_file(self, json_path: Path):
            """Load settings from metadata JSON."""
            try:
                metadata = load_mlm_metadata(json_path)
                
                # Check tool type
                tool = metadata.get("tool", "unknown")
                settings = metadata.get("settings", {})
                
                if tool == "RoBERTaMaskedLanguageModelVerbs":
                    self.input_text.setText(metadata.get("input_file", ""))
                    self.output_text.setText(metadata.get("output_file", ""))
                    self.model_combo.setCurrentText(settings.get("model", "roberta-base"))
                    self.batch_spin.setValue(settings.get("batch_size", 16))
                    self.topk_spin.setValue(settings.get("top_k", 10))
                    
                    device = settings.get("device", "auto")
                    if "cuda" in str(device).lower():
                        self.device_combo.setCurrentText("cuda")
                    elif "mps" in str(device).lower():
                        self.device_combo.setCurrentText("mps")
                    elif "cpu" in str(device).lower():
                        self.device_combo.setCurrentText("cpu")
                    else:
                        self.device_combo.setCurrentText("auto")
                elif tool == "SpaCyVerbExtractor" or tool == "FilterSpaCyVerbs":
                    # Use extractor/filter output as input and default output in same folder
                    input_csv = metadata.get("output_file", "")
                    self.input_text.setText(input_csv)
                    if input_csv:
                        in_path = Path(input_csv)
                        default_out = in_path.with_name(f"{in_path.stem}.mlm.csv")
                        self.output_text.setText(str(default_out))
                else:
                    QMessageBox.warning(
                        self,
                        "Unknown Metadata Type",
                        f"This metadata is from '{tool}'. Expected RoBERTaMaskedLanguageModelVerbs, SpaCyVerbExtractor, or FilterSpaCyVerbs."
                    )
                    return
                
                # Verify checksum when input is set
                if self.input_text.text():
                    matches, message = verify_input_checksum(Path(self.input_text.text()), metadata)
                    self.log(message)
                
                self.log(f"✓ Loaded settings from: {json_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load metadata: {str(e)}")
        
        def log(self, message: str):
            """Add message to log."""
            self.log_text.append(message)
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )
        
        def start_inference(self):
            """Start the MLM inference process."""
            if not self.input_text.text():
                QMessageBox.warning(self, "Missing Input", "Please select an input CSV file")
                return
            
            if not self.output_text.text():
                QMessageBox.warning(self, "Missing Output", "Please select an output file")
                return
            
            input_path = Path(self.input_text.text()).resolve()
            output_path = Path(self.output_text.text()).resolve()
            
            if not input_path.exists():
                QMessageBox.critical(self, "Error", f"Input file not found: {input_path}")
                return
            
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot create output directory: {str(e)}")
                return
            
            # Check for metadata warnings before starting
            metadata_path = output_path.with_suffix(".json")
            if metadata_path.exists():
                try:
                    metadata = load_mlm_metadata(metadata_path)
                    matches, message = verify_input_checksum(input_path, metadata)
                    if not matches:
                        msg = message + "\n\nContinue with MLM inference?"
                        reply = QMessageBox.warning(self, "Checksum Mismatch", msg, QMessageBox.Yes | QMessageBox.No)
                        if reply == QMessageBox.No:
                            return
                except:
                    pass
            
            # Clear log
            self.log_text.clear()
            self.log("=" * 60)
            self.log("Starting MLM Inference")
            self.log("=" * 60)
            self.log(f"Input: {input_path}")
            self.log(f"Output: {output_path}")
            self.log(f"Model: {self.model_combo.currentText()}")
            self.log(f"Batch size: {self.batch_spin.value()}")
            self.log(f"Top-K: {self.topk_spin.value()}")
            self.log(f"Device: {self.device_combo.currentText()}")
            if self.debug_check.isChecked():
                self.log(f"Debug limit: {self.debug_spin.value()}")
            self.log("")
            
            # Disable controls
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.input_browse.setEnabled(False)
            self.output_browse.setEnabled(False)
            
            # Start worker
            self.worker = MLMWorker(
                input_path,
                output_path,
                self.model_combo.currentText(),
                self.batch_spin.value(),
                self.topk_spin.value(),
                self.device_combo.currentText(),
                self.debug_spin.value() if self.debug_check.isChecked() else 0,
            )
            
            self.worker.progress_update.connect(self.log)
            self.worker.finished.connect(self.on_finished)
            self.worker.start()
        
        def stop_inference(self):
            """Stop the MLM inference process."""
            if self.worker:
                self.log("Stopping...")
                self.worker.request_stop()
                self.worker.wait()
                self.on_finished(False, "MLM inference stopped by user")
        
        def on_finished(self, success: bool, message: str):
            """Handle MLM inference completion."""
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.input_browse.setEnabled(True)
            self.output_browse.setEnabled(True)
            
            self.log("")
            self.log("=" * 60)
            self.log(message)
            self.log("=" * 60)
            
            if success:
                QMessageBox.information(self, "Success", message)
            else:
                QMessageBox.critical(self, "Error", message)

    app = QApplication(sys.argv)
    window = MLMInferenceGUI()
    window.show()
    sys.exit(app.exec())


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    """Main entry point - choose GUI or CLI mode."""
    if len(sys.argv) == 1:
        print("Launching GUI mode...")
        print("(Use command-line arguments to run in CLI mode. Use --help for usage.)")
        run_gui()
    else:
        run_cli()


if __name__ == "__main__":
    main()