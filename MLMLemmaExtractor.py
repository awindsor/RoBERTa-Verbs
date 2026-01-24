#!/usr/bin/env python3
"""
MLM Lemma Extractor - Extract and explore sentences by lemma from MLM CSV output.

Supports both CLI and GUI modes:
  - Run without arguments: launches GUI (requires PySide6)
  - Run with arguments: uses CLI mode

This tool searches MLM output CSVs for specific lemmas and displays matching
sentences with optional top-k predictions.

Requirements:
  pip install PySide6  # for GUI mode only

CLI Examples:
  python MLMLemmaExtractor.py run mlm_out.csv
  python MLMLemmaExtractor.py run mlm_out.csv --limit 10
  python MLMLemmaExtractor.py run mlm_out.csv --show-topk
  python MLMLemmaExtractor.py run mlm_out.csv --show-topk --top-k 5
  python MLMLemmaExtractor.py run,walk,think mlm_out.csv --show-topk

GUI Mode:
  python MLMLemmaExtractor.py
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

PROB_COL_RE = re.compile(r"^prob_(\d+)$")


PROB_COL_RE = re.compile(r"^prob_(\d+)$")


# ============================================================================
# CORE LOGIC (shared by CLI and GUI)
# ============================================================================

def infer_top_k(fieldnames: List[str]) -> int:
    """Infer top-k value from CSV header."""
    k = 0
    for name in fieldnames:
        m = PROB_COL_RE.match(name.strip())
        if m:
            k = max(k, int(m.group(1)))
    return k


def extract_lemma_sentences(
    mlm_csv_path: Path,
    lemmas: List[str],
    lemma_col: str = "lemma",
    sentence_col: str = "sentence",
    limit: int = 0,
    show_topk: bool = False,
    top_k: int = 0,
    encoding: str = "utf-8",
) -> Tuple[List[Tuple[str, str, List[Tuple[str, str]]]], int]:
    """
    Extract sentences matching given lemmas from MLM CSV.
    
    Returns:
        Tuple of (results, actual_k) where:
        - results: List of tuples: (lemma, sentence, predictions)
          where predictions is [(token, prob), ...]
        - actual_k: The actual top-k value available in the CSV
    """
    results: List[Tuple[str, str, List[Tuple[str, str]]]] = []
    target_lemmas = {lemma.strip().lower() for lemma in lemmas}
    count = 0
    
    with mlm_csv_path.open(newline="", encoding=encoding) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")
        
        if lemma_col not in reader.fieldnames or sentence_col not in reader.fieldnames:
            raise ValueError(f"Missing required columns: {lemma_col} or {sentence_col}")
        
        # Infer actual top-k from CSV header
        actual_k = infer_top_k(list(reader.fieldnames))
        
        # Use the lesser of requested top_k and actual available top_k
        k = min(top_k, actual_k) if top_k > 0 else actual_k
        
        for row in reader:
            lemma = (row.get(lemma_col) or "").strip().lower()
            if lemma not in target_lemmas:
                continue
            
            sentence = row[sentence_col]
            
            preds: List[Tuple[str, str]] = []
            if show_topk:
                for i in range(1, k + 1):
                    tok = row.get(f"token_{i}")
                    prob = row.get(f"prob_{i}")
                    if tok and prob:
                        preds.append((tok, prob))
            
            results.append((lemma, sentence, preds))
            
            count += 1
            if limit and count >= limit:
                break
    
    return results, actual_k


# ============================================================================
# CLI MODE
# ============================================================================

def run_cli() -> None:
    """Run in CLI mode."""
    ap = argparse.ArgumentParser(
        description="MLM Lemma Extractor - Extract sentences by lemma from MLM CSV"
    )
    ap.add_argument("lemmas", help="Target lemma(s) to match (comma-separated for multiple)")
    ap.add_argument("mlm_csv", help="MLM output CSV")
    ap.add_argument("--lemma-col", default="lemma", help="Lemma column name (default: lemma)")
    ap.add_argument("--sentence-col", default="sentence", help="Sentence column name (default: sentence)")
    ap.add_argument("--limit", type=int, default=0, help="Max number of sentences (0 = no limit)")
    ap.add_argument(
        "--show-topk",
        action="store_true",
        help="Print stored top-k predictions under each sentence",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="How many predictions to show (default: infer from header)",
    )
    ap.add_argument("--encoding", default="utf-8", help="CSV encoding (default: utf-8)")
    args = ap.parse_args()
    
    # Parse lemmas (comma-separated)
    lemmas = [l.strip() for l in args.lemmas.split(",") if l.strip()]
    if not lemmas:
        raise SystemExit("Error: No lemmas provided")
    
    mlm_path = Path(args.mlm_csv)
    if not mlm_path.exists():
        raise SystemExit(f"Error: File not found: {mlm_path}")
    
    try:
        results, actual_k = extract_lemma_sentences(
            mlm_path,
            lemmas,
            args.lemma_col,
            args.sentence_col,
            args.limit,
            args.show_topk,
            args.top_k,
            args.encoding,
        )
    except Exception as e:
        raise SystemExit(f"Error: {e}")
    
    if not results:
        print(f"No matches found for lemma(s): {', '.join(lemmas)}")
        return
    
    # Warn if requested top-k exceeds available
    if args.top_k > 0 and args.top_k > actual_k:
        print(f"⚠ Warning: Requested top-k ({args.top_k}) exceeds available predictions ({actual_k})")
        print(f"⚠ Displaying {actual_k} predictions per sentence\n")
    
    # Print results
    for lemma, sentence, preds in results:
        print(sentence)
        
        if args.show_topk and preds:
            for tok, prob in preds:
                print(f"    {tok:>15s}  {prob}")
        elif args.show_topk:
            print("    [no top-k predictions found]")
        
        print()  # blank line between hits
    
    print(f"\n--- Found {len(results)} sentence(s) ---")


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
            QMessageBox,
        )
    except ImportError:
        print("Error: PySide6 is required for GUI mode.")
        print("Install it with: pip install PySide6")
        print("\nOr run in CLI mode by providing arguments. Use --help for usage.")
        sys.exit(1)
    
    class SearchWorker(QThread):
        """Worker thread for searching MLM CSV without blocking UI."""
        
        progress_update = Signal(str)
        finished = Signal(bool, str, list, int)
        
        def __init__(
            self,
            mlm_csv_path: Path,
            lemmas: List[str],
            lemma_col: str,
            sentence_col: str,
            limit: int,
            show_topk: bool,
            top_k: int,
            encoding: str,
        ):
            super().__init__()
            self.mlm_csv_path = mlm_csv_path
            self.lemmas = lemmas
            self.lemma_col = lemma_col
            self.sentence_col = sentence_col
            self.limit = limit
            self.show_topk = show_topk
            self.top_k = top_k
            self.encoding = encoding
        
        def run(self):
            """Run the search in the worker thread."""
            try:
                self.progress_update.emit(f"Searching for lemma(s): {', '.join(self.lemmas)}")
                self.progress_update.emit(f"Reading: {self.mlm_csv_path}")
                
                results, actual_k = extract_lemma_sentences(
                    self.mlm_csv_path,
                    self.lemmas,
                    self.lemma_col,
                    self.sentence_col,
                    self.limit,
                    self.show_topk,
                    self.top_k,
                    self.encoding,
                )
                
                # Warn if requested top-k exceeds available
                if self.top_k > 0 and self.top_k > actual_k:
                    self.progress_update.emit(
                        f"⚠ Warning: Requested top-k ({self.top_k}) exceeds available ({actual_k})"
                    )
                    self.progress_update.emit(f"⚠ Displaying {actual_k} predictions per sentence")
                
                self.progress_update.emit(f"Found {len(results)} sentence(s)")
                self.finished.emit(True, f"Search complete. Found {len(results)} sentence(s).", results, actual_k)
                
            except Exception as e:
                self.progress_update.emit(f"✗ Error: {str(e)}")
                self.finished.emit(False, f"Error: {str(e)}", [], 0)
    
    class MLMLemmaExtractorGUI(QMainWindow):
        """Main GUI window for MLM lemma extraction."""
        
        def __init__(self):
            super().__init__()
            self.setWindowTitle("MLM Lemma Extractor")
            self.setMinimumSize(QSize(900, 700))
            
            self.worker: Optional[SearchWorker] = None
            
            self.init_ui()
        
        def init_ui(self):
            """Initialize the UI components."""
            central = QWidget()
            self.setCentralWidget(central)
            
            layout = QVBoxLayout(central)
            
            # File section
            file_group = QGroupBox("MLM CSV File")
            file_layout = QHBoxLayout()
            
            self.file_input = QLineEdit()
            self.file_browse = QPushButton("Browse...")
            self.file_browse.clicked.connect(self.browse_file)
            
            file_layout.addWidget(QLabel("CSV File:"))
            file_layout.addWidget(self.file_input)
            file_layout.addWidget(self.file_browse)
            
            file_group.setLayout(file_layout)
            layout.addWidget(file_group)
            
            # Lemma section
            lemma_group = QGroupBox("Lemmas to Search")
            lemma_layout = QVBoxLayout()
            
            lemma_help = QLabel("Enter lemmas (one per line or comma-separated):")
            lemma_help.setStyleSheet("font-style: italic; color: #666;")
            lemma_layout.addWidget(lemma_help)
            
            self.lemma_input = QTextEdit()
            self.lemma_input.setMaximumHeight(100)
            self.lemma_input.setPlaceholderText("run\nwalk\nthink")
            lemma_layout.addWidget(self.lemma_input)
            
            lemma_group.setLayout(lemma_layout)
            layout.addWidget(lemma_group)
            
            # Options section
            options_group = QGroupBox("Options")
            options_layout = QVBoxLayout()
            
            # Column names
            col_layout = QHBoxLayout()
            
            lemma_col_layout = QVBoxLayout()
            lemma_col_layout.addWidget(QLabel("Lemma Column:"))
            self.lemma_col_input = QLineEdit("lemma")
            lemma_col_layout.addWidget(self.lemma_col_input)
            
            sent_col_layout = QVBoxLayout()
            sent_col_layout.addWidget(QLabel("Sentence Column:"))
            self.sent_col_input = QLineEdit("sentence")
            sent_col_layout.addWidget(self.sent_col_input)
            
            col_layout.addLayout(lemma_col_layout)
            col_layout.addLayout(sent_col_layout)
            options_layout.addLayout(col_layout)
            
            # Show top-k and limit
            settings_layout = QHBoxLayout()
            
            self.show_topk_check = QCheckBox("Show Top-K Predictions")
            settings_layout.addWidget(self.show_topk_check)
            
            settings_layout.addWidget(QLabel("Top-K:"))
            self.topk_spin = QSpinBox()
            self.topk_spin.setMinimum(0)
            self.topk_spin.setMaximum(100)
            self.topk_spin.setValue(0)
            self.topk_spin.setToolTip("0 = use all available predictions from CSV")
            settings_layout.addWidget(self.topk_spin)
            
            self.topk_info = QLabel("(will be inferred from CSV)")
            self.topk_info.setStyleSheet("font-style: italic; color: #666;")
            settings_layout.addWidget(self.topk_info)
            
            settings_layout.addWidget(QLabel("Limit:"))
            self.limit_spin = QSpinBox()
            self.limit_spin.setMinimum(0)
            self.limit_spin.setMaximum(100000)
            self.limit_spin.setValue(0)
            self.limit_spin.setToolTip("0 = no limit")
            settings_layout.addWidget(self.limit_spin)
            
            settings_layout.addStretch()
            options_layout.addLayout(settings_layout)
            
            options_group.setLayout(options_layout)
            layout.addWidget(options_group)
            
            # Results section
            results_group = QGroupBox("Results")
            results_layout = QVBoxLayout()
            
            self.results_text = QTextEdit()
            self.results_text.setReadOnly(True)
            self.results_text.setFont(self.results_text.font())
            results_layout.addWidget(self.results_text)
            
            results_group.setLayout(results_layout)
            layout.addWidget(results_group)
            
            # Control buttons
            button_layout = QHBoxLayout()
            
            self.search_btn = QPushButton("Search")
            self.search_btn.clicked.connect(self.start_search)
            self.search_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
            
            self.clear_btn = QPushButton("Clear Results")
            self.clear_btn.clicked.connect(self.clear_results)
            
            button_layout.addStretch()
            button_layout.addWidget(self.clear_btn)
            button_layout.addWidget(self.search_btn)
            button_layout.addStretch()
            
            layout.addLayout(button_layout)
            
            central.setLayout(layout)
        
        def browse_file(self):
            """Open file dialog for MLM CSV file."""
            file, _ = QFileDialog.getOpenFileName(
                self,
                "Select MLM CSV File",
                "",
                "CSV Files (*.csv);;All Files (*)"
            )
            if file:
                self.file_input.setText(file)
        
        def clear_results(self):
            """Clear the results text area."""
            self.results_text.clear()
        
        def log(self, message: str):
            """Add message to results."""
            self.results_text.append(message)
            self.results_text.verticalScrollBar().setValue(
                self.results_text.verticalScrollBar().maximum()
            )
        
        def start_search(self):
            """Start the search process."""
            if not self.file_input.text():
                QMessageBox.warning(self, "Missing File", "Please select an MLM CSV file")
                return
            
            # Parse lemmas
            lemma_text = self.lemma_input.toPlainText().strip()
            if not lemma_text:
                QMessageBox.warning(self, "Missing Lemmas", "Please enter at least one lemma to search")
                return
            
            # Support both newline and comma separation
            lemmas = []
            for line in lemma_text.split('\n'):
                for lemma in line.split(','):
                    lemma = lemma.strip()
                    if lemma:
                        lemmas.append(lemma)
            
            if not lemmas:
                QMessageBox.warning(self, "Missing Lemmas", "Please enter at least one lemma to search")
                return
            
            mlm_path = Path(self.file_input.text()).resolve()
            if not mlm_path.exists():
                QMessageBox.critical(self, "Error", f"File not found: {mlm_path}")
                return
            
            # Clear previous results
            self.results_text.clear()
            self.log("=" * 60)
            self.log("MLM Lemma Search")
            self.log("=" * 60)
            self.log(f"File: {mlm_path}")
            self.log(f"Searching for: {', '.join(lemmas)}")
            self.log("")
            
            # Disable search button
            self.search_btn.setEnabled(False)
            
            # Start worker
            self.worker = SearchWorker(
                mlm_path,
                lemmas,
                self.lemma_col_input.text(),
                self.sent_col_input.text(),
                self.limit_spin.value(),
                self.show_topk_check.isChecked(),
                self.topk_spin.value(),
                "utf-8",
            )
            
            self.worker.progress_update.connect(self.log)
            self.worker.finished.connect(self.on_finished)
            self.worker.start()
        
        def format_predictions_table(self, preds: list) -> str:
            """Format predictions as an ASCII table."""
            if not preds:
                return "    [no top-k predictions found]"
            
            # Calculate column widths
            max_token_len = max(len(tok) for tok, _ in preds) if preds else 5
            max_prob_len = max(len(prob) for _, prob in preds) if preds else 10
            
            token_width = max(max_token_len, 5)
            prob_width = max(max_prob_len, 11)
            
            lines = []
            
            # Header
            header = f"    {'Token':<{token_width}}  {'Probability':>{prob_width}}"
            lines.append(header)
            lines.append(f"    {'-' * token_width}  {'-' * prob_width}")
            
            # Data rows
            for tok, prob in preds:
                lines.append(f"    {tok:<{token_width}}  {prob:>{prob_width}}")
            
            return "\n".join(lines)
        
        def on_finished(self, success: bool, message: str, results: list, actual_k: int):
            """Handle search completion."""
            self.search_btn.setEnabled(True)
            
            # Update the info label with actual top-k from CSV
            if actual_k > 0:
                self.topk_info.setText(f"(CSV has {actual_k} predictions per row)")
                # Update spinbox maximum to match available predictions
                self.topk_spin.setMaximum(max(100, actual_k))
            
            self.log("")
            self.log("=" * 60)
            
            if success and results:
                self.log(f"Found {len(results)} sentence(s):\n")
                
                # Group results by lemma
                by_lemma: dict[str, list[tuple[str, list[tuple[str, str]]]]] = {}
                for lemma, sentence, preds in results:
                    if lemma not in by_lemma:
                        by_lemma[lemma] = []
                    by_lemma[lemma].append((sentence, preds))
                
                # Display results grouped by lemma
                for lemma in sorted(by_lemma.keys()):
                    self.log(f"▸ Lemma: {lemma.upper()} ({len(by_lemma[lemma])} sentence(s))")
                    self.log("-" * 60)
                    
                    for sentence, preds in by_lemma[lemma]:
                        self.log(sentence)
                        
                        if self.show_topk_check.isChecked():
                            table = self.format_predictions_table(preds)
                            self.log(table)
                        
                        self.log("")  # blank line between sentences
                    
                    self.log("")  # blank line between lemmas
            elif success:
                self.log("No matching sentences found.")
            else:
                self.log(f"Error: {message}")
            
            self.log("=" * 60)
    
    app = QApplication(sys.argv)
    window = MLMLemmaExtractorGUI()
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