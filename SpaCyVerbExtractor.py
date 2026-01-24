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
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import spacy


# ============================================================================
# CORE EXTRACTION LOGIC (shared by CLI and GUI)
# ============================================================================

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
    t0 = time.time()
    logger.info(f"Loading spaCy model: {args.model}")
    nlp = spacy.load(args.model, disable=["ner", "textcat"])
    if "parser" not in nlp.pipe_names:
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

            try:
                file_size_bytes = doc_path.stat().st_size
            except OSError:
                file_size_bytes = 0

            doc_start = time.time()

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

    logger.info(f"Wrote output: {out_path}")


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
        
        def request_stop(self):
            """Request graceful stop."""
            self._stop_requested = True
        
        def run(self):
            """Run the extraction in the worker thread."""
            try:
                # Load spaCy model
                self.progress_update.emit(f"Loading spaCy model: {self.model}")
                nlp = spacy.load(self.model, disable=["ner", "textcat"])
                if "parser" not in nlp.pipe_names:
                    if "sentencizer" not in nlp.pipe_names:
                        nlp.add_pipe("sentencizer")
                self.progress_update.emit(f"spaCy pipeline: {nlp.pipe_names}")
                
                delimiter = "\t" if self.use_tsv else ","
                
                overall_verbs = 0
                overall_sents = 0
                overall_chunks = 0
                
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
                    
                    for doc_path in self.paths:
                        if self._stop_requested:
                            break
                        
                        if not doc_path.exists():
                            self.progress_update.emit(f"⚠ Skipping missing path: {doc_path}")
                            continue
                        
                        self.progress_update.emit(f"Starting document: {doc_path}")
                        
                        try:
                            file_size_bytes = doc_path.stat().st_size
                        except OSError:
                            file_size_bytes = 0
                        
                        seen: Dict[Tuple[int, str], int] = {}
                        order: List[Tuple[int, str]] = []
                        sent_counter = 0
                        verb_counter = 0
                        chunk_counter = 0
                        
                        for chunk_start, chunk_text in stream_char_chunks(
                            doc_path,
                            encoding=self.encoding,
                            chunk_size=self.chunk_size,
                            overlap=self.overlap,
                            stop_check=lambda: self._stop_requested,
                        ):
                            if self._stop_requested:
                                break
                            
                            chunk_counter += 1
                            overall_chunks += 1
                            chunk_end = chunk_start + len(chunk_text)
                            
                            if file_size_bytes > 0:
                                pct = min(100.0, (chunk_end / file_size_bytes) * 100.0)
                            else:
                                pct = 0.0
                            
                            self.progress_update.emit(
                                f"  Chunk {chunk_counter:6d} | chars {chunk_start:,}–{chunk_end:,} | ~{pct:6.2f}%"
                            )
                            
                            doc = nlp(chunk_text)
                            
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
                            
                            self.chunk_count_update.emit(overall_chunks)
                        
                        overall_sents += sent_counter
                        overall_verbs += verb_counter
                        
                        self.progress_update.emit(
                            f"Finished document: {doc_path} | "
                            f"{chunk_counter:,} chunks | {sent_counter:,} sentences | {verb_counter:,} verbs"
                        )
                
                self.progress_update.emit(
                    f"✓ Run complete | {overall_chunks:,} chunks | {overall_sents:,} sentences | {overall_verbs:,} verbs"
                )
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
            self.progress_bar.setVisible(False)
            progress_layout.addWidget(self.progress_bar)
            
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
        
        def log(self, message: str):
            """Append a message to the log output."""
            self.log_text.append(message)
            # Auto-scroll to bottom
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )
        
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
            
            # Clear log
            self.log_text.clear()
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
