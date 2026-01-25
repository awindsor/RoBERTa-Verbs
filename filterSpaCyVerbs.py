#!/usr/bin/env python3
"""
Filter a verb CSV by frequency of a chosen field with CLI + GUI + metadata support.

Supports both CLI and GUI modes:
  - Run without arguments: launches GUI (requires PySide6)
  - Run with arguments: uses CLI mode

Two-pass, streaming-friendly filtering:
  Pass 1: count frequencies of the chosen field (lemma or surface_lower)
  Pass 2: write only rows whose field frequency is within [min_freq, max_freq] (inclusive)

Can load settings from:
  - filterSpaCyVerbs JSON metadata (from previous filterSpaCyVerbs run)
  - SpaCyVerbExtractor JSON metadata (to filter extracted verbs)

Output includes JSON metadata with:
  - All filtering settings
  - MD5 checksums of input/output
  - Filter statistics

Requirements:
  pip install PySide6  # for GUI mode only

CLI Examples:
  python filterSpaCyVerbs.py verbs.csv output.csv --field lemma --min-freq 10
  python filterSpaCyVerbs.py verbs.csv output.csv --field lemma --min-freq 10 --max-freq 1000
  python filterSpaCyVerbs.py --load-metadata verbs.json verbs.csv output.csv

GUI Mode:
  python filterSpaCyVerbs.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ============================================================================
# CORE FILTERING LOGIC (shared by CLI and GUI)
# ============================================================================

def compute_file_md5(path: Path) -> str:
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def save_filter_metadata(
    output_path: Path,
    input_path: Path,
    input_checksum: str,
    output_checksum: str,
    args: Dict,
    stats: Dict,
) -> None:
    """Save filtering metadata to JSON file alongside output."""
    json_path = output_path.with_suffix(".json")
    
    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tool": "filterSpaCyVerbs",
        "input_file": str(input_path),
        "input_checksum": input_checksum,
        "output_file": str(output_path),
        "output_checksum": output_checksum,
        "settings": {
            "field": args.get("field", "lemma"),
            "min_freq": args.get("min_freq"),
            "max_freq": args.get("max_freq"),
        },
        "statistics": stats,
    }
    
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_filter_metadata(json_path: Path) -> dict[Any, Any]:
    """Load filtering metadata from JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        return dict(json.load(f))


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


def count_field_freq(path: Path, field: str) -> Counter:
    """Count frequencies of field values."""
    counts: Counter = Counter()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or field not in reader.fieldnames:
            raise ValueError(f"Field '{field}' not found in input header: {reader.fieldnames}")
        for row in reader:
            counts[row[field]] += 1
    return counts


def in_range(freq: int, min_freq: Optional[int], max_freq: Optional[int]) -> bool:
    """Check if frequency falls within range."""
    if min_freq is not None and freq < min_freq:
        return False
    if max_freq is not None and freq > max_freq:
        return False
    return True


def count_input_rows(input_path: Path) -> int:
    """Count total data rows in input CSV (excluding header)."""
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def filter_rows(
    input_path: Path,
    output_path: Path,
    field: str,
    counts: Counter,
    min_freq: Optional[int],
    max_freq: Optional[int],
) -> int:
    """Filter rows and write to output. Returns count of rows written."""
    rows_written = 0
    with input_path.open(newline="", encoding="utf-8") as fin, output_path.open(
        "w", newline="", encoding="utf-8"
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
                rows_written += 1
    
    return rows_written


# ============================================================================
# CLI MODE
# ============================================================================

def run_cli() -> None:
    """Run in CLI mode."""
    ap = argparse.ArgumentParser(
        description="Filter verb CSV by frequency thresholds"
    )
    ap.add_argument("input_csv", nargs="?", help="Input verb CSV file")
    ap.add_argument("output_csv", nargs="?", help="Output filtered CSV file")
    ap.add_argument("--field", choices=["lemma", "surface_lower"], help="Field to filter by")
    ap.add_argument("--min-freq", type=int, help="Keep values with freq >= min_freq")
    ap.add_argument("--max-freq", type=int, help="Keep values with freq <= max_freq")
    ap.add_argument("--load-metadata", help="Load settings from metadata JSON (CLI args override)")
    args = ap.parse_args()

    # Load metadata if provided
    metadata = None
    if args.load_metadata:
        metadata_path = Path(args.load_metadata)
        if not metadata_path.exists():
            raise SystemExit(f"Metadata file not found: {metadata_path}")
        
        metadata = load_filter_metadata(metadata_path)
        
        # Use loaded settings as defaults, CLI args override
        if not args.field:
            args.field = metadata.get("settings", {}).get("field", "lemma")
        if args.min_freq is None:
            args.min_freq = metadata.get("settings", {}).get("min_freq")
        if args.max_freq is None:
            args.max_freq = metadata.get("settings", {}).get("max_freq")
        
        # If input not provided on CLI, use from metadata
        if not args.input_csv:
            args.input_csv = metadata.get("input_file")
    
    # Validate required arguments
    if not args.input_csv or not args.output_csv:
        ap.print_help()
        raise SystemExit("Error: input_csv and output_csv are required (or use --load-metadata)")
    
    if not args.field:
        raise SystemExit("Error: --field is required")
    
    if args.min_freq is None and args.max_freq is None:
        raise SystemExit("Error: At least one of --min-freq or --max-freq must be specified")
    
    if args.min_freq is not None and args.min_freq < 1:
        raise SystemExit("Error: --min-freq must be >= 1")
    if args.max_freq is not None and args.max_freq < 1:
        raise SystemExit("Error: --max-freq must be >= 1")
    if args.min_freq is not None and args.max_freq is not None and args.min_freq > args.max_freq:
        raise SystemExit("Error: --min-freq cannot be greater than --max-freq")
    
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    
    # Verify checksum if metadata was loaded
    if metadata:
        matches, message = verify_input_checksum(input_path, metadata)
        if not matches:
            print(f"⚠ {message}")
    
    # Filter
    print(f"Counting frequencies for field '{args.field}'...")
    counts = count_field_freq(input_path, args.field)
    
    print("Counting input rows...")
    total_input_rows = count_input_rows(input_path)
    
    print("Filtering and writing output...")
    rows_written = filter_rows(
        input_path,
        output_path,
        args.field,
        counts,
        args.min_freq,
        args.max_freq,
    )
    
    rows_filtered = total_input_rows - rows_written
    
    # Compute checksums and save metadata
    input_checksum = compute_file_md5(input_path)
    output_checksum = compute_file_md5(output_path)
    
    filter_args = {
        "field": args.field,
        "min_freq": args.min_freq,
        "max_freq": args.max_freq,
    }
    
    stats = {
        "rows_written": rows_written,
        "rows_filtered_out": rows_filtered,
        "total_input_rows": total_input_rows,
        "unique_values": len(counts),
    }
    
    save_filter_metadata(output_path, input_path, input_checksum, output_checksum, filter_args, stats)
    
    print(f"✓ Wrote {rows_written} rows to {output_path}")
    print(f"✓ Filtered out {rows_filtered} rows")
    print(f"✓ Saved metadata to {output_path.with_suffix('.json')}")


# ============================================================================
# GUI MODE
# ============================================================================

def run_gui() -> None:
    """Launch the GUI application."""
    try:
        from PySide6.QtCore import QThread, Signal, QSize
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
        )
    except ImportError:
        print("Error: PySide6 is required for GUI mode.")
        print("Install it with: pip install PySide6\n")
        print("Or run in CLI mode by providing arguments. Use --help for usage.")
        sys.exit(1)

    class FilterWorker(QThread):
        """Worker thread for filtering without blocking UI."""
        
        progress_update = Signal(str)
        stats_update = Signal(str, str)
        finished = Signal(bool, str)
        
        def __init__(
            self,
            input_path: Path,
            output_path: Path,
            field: str,
            min_freq: Optional[int],
            max_freq: Optional[int],
        ):
            super().__init__()
            self.input_path = input_path
            self.output_path = output_path
            self.field = field
            self.min_freq = min_freq
            self.max_freq = max_freq
            self._stop_requested = False
        
        def request_stop(self):
            """Request graceful stop."""
            self._stop_requested = True
        
        def run(self):
            """Run the filtering in the worker thread."""
            try:
                self.progress_update.emit(f"Counting frequencies for field '{self.field}'...")
                counts = count_field_freq(self.input_path, self.field)
                
                self.progress_update.emit(f"Found {len(counts)} unique values")
                self.progress_update.emit("Counting input rows...")
                total_input_rows = count_input_rows(self.input_path)
                self.progress_update.emit(f"Total input rows: {total_input_rows}")
                
                self.progress_update.emit("Filtering and writing output...")
                
                rows_written = filter_rows(
                    self.input_path,
                    self.output_path,
                    self.field,
                    counts,
                    self.min_freq,
                    self.max_freq,
                )
                
                rows_filtered = total_input_rows - rows_written
                
                self.progress_update.emit("Writing metadata...")
                input_checksum = compute_file_md5(self.input_path)
                output_checksum = compute_file_md5(self.output_path)
                
                filter_args = {
                    "field": self.field,
                    "min_freq": self.min_freq,
                    "max_freq": self.max_freq,
                }
                
                stats = {
                    "rows_written": rows_written,
                    "rows_filtered_out": rows_filtered,
                    "total_input_rows": total_input_rows,
                    "unique_values": len(counts),
                }
                
                save_filter_metadata(self.output_path, self.input_path, input_checksum, output_checksum, filter_args, stats)
                
                self.progress_update.emit(f"✓ Wrote {rows_written} rows")
                self.progress_update.emit(f"✓ Filtered out {rows_filtered} rows")
                self.progress_update.emit(f"✓ Saved metadata: {self.output_path.with_suffix('.json')}")
                self.finished.emit(True, f"Filtering complete. Output: {self.output_path}")
                
            except Exception as e:
                self.progress_update.emit(f"✗ Error: {str(e)}")
                self.finished.emit(False, f"Error: {str(e)}")

    class FilterSpaCyVerbsGUI(QMainWindow):
        """Main GUI window for filter tool."""
        
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Filter SpaCy Verbs")
            self.setMinimumSize(QSize(900, 600))
            
            self.worker: Optional[FilterWorker] = None
            
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
            self.output_text = QLineEdit("filtered_verbs.csv")
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
            
            # Field selection
            field_layout = QHBoxLayout()
            field_label = QLabel("Field to Filter:")
            self.field_combo = QComboBox()
            self.field_combo.addItems(["lemma", "surface_lower"])
            field_layout.addWidget(field_label)
            field_layout.addWidget(self.field_combo)
            field_layout.addStretch()
            settings_layout.addLayout(field_layout)
            
            # Frequency range
            freq_layout = QHBoxLayout()
            
            min_layout = QVBoxLayout()
            min_label = QLabel("Min Frequency:")
            self.min_spin = QSpinBox()
            self.min_spin.setMinimum(0)
            self.min_spin.setMaximum(1000000)
            self.min_spin.setValue(1)
            min_layout.addWidget(min_label)
            min_layout.addWidget(self.min_spin)
            
            max_layout = QVBoxLayout()
            max_label = QLabel("Max Frequency:")
            self.max_spin = QSpinBox()
            self.max_spin.setMinimum(0)
            self.max_spin.setMaximum(10000000)
            self.max_spin.setValue(0)  # 0 = no limit
            max_layout.addWidget(max_label)
            max_layout.addWidget(self.max_spin)
            
            freq_layout.addLayout(min_layout)
            freq_layout.addLayout(max_layout)
            freq_layout.addStretch()
            settings_layout.addLayout(freq_layout)
            
            settings_group.setLayout(settings_layout)
            layout.addWidget(settings_group)
            
            # Progress section
            progress_group = QGroupBox("Progress")
            progress_layout = QVBoxLayout()
            
            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            self.log_text.setMaximumHeight(150)
            progress_layout.addWidget(self.log_text)
            
            progress_group.setLayout(progress_layout)
            layout.addWidget(progress_group)
            
            # Control buttons
            button_layout = QHBoxLayout()
            self.start_btn = QPushButton("Start Filtering")
            self.start_btn.clicked.connect(self.start_filtering)
            self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
            self.stop_btn = QPushButton("Stop")
            self.stop_btn.clicked.connect(self.stop_filtering)
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
                "filtered_verbs.csv",
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
                metadata = load_filter_metadata(json_path)
                
                # Handle both filterSpaCyVerbs and SpaCyVerbExtractor metadata
                tool = metadata.get("tool", "unknown")
                
                if tool == "filterSpaCyVerbs":
                    # Filter metadata
                    settings = metadata.get("settings", {})
                    self.input_text.setText(metadata.get("input_file", ""))
                    self.output_text.setText(metadata.get("output_file", ""))
                    self.field_combo.setCurrentText(settings.get("field", "lemma"))
                    self.min_spin.setValue(settings.get("min_freq") or 1)
                    self.max_spin.setValue(settings.get("max_freq") or 0)
                    self.log(f"✓ Loaded filter settings from: {json_path}")
                
                elif tool == "SpaCyVerbExtractor" or metadata.get("output_file", "").endswith(".csv"):
                    # Extractor metadata - use as input
                    self.input_text.setText(metadata.get("output_file", ""))
                    
                    # Verify checksum
                    matches, message = verify_input_checksum(Path(self.input_text.text()), metadata)
                    self.log(message)
                    
                    self.log(f"✓ Loaded from extractor output: {json_path}")
                else:
                    raise ValueError(f"Unknown metadata format (tool={tool})")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load metadata: {str(e)}")
        
        def log(self, message: str):
            """Add message to log."""
            self.log_text.append(message)
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )
        
        def start_filtering(self):
            """Start the filtering process."""
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
                    metadata = load_filter_metadata(metadata_path)
                    matches, message = verify_input_checksum(input_path, metadata)
                    if not matches:
                        msg = message + "\n\nContinue with filtering?"
                        reply = QMessageBox.warning(self, "Checksum Mismatch", msg, QMessageBox.Yes | QMessageBox.No)
                        if reply == QMessageBox.No:
                            return
                except Exception:
                    pass
            
            # Clear log
            self.log_text.clear()
            self.log("=" * 60)
            self.log("Starting Filtering")
            self.log("=" * 60)
            self.log(f"Input: {input_path}")
            self.log(f"Output: {output_path}")
            self.log(f"Field: {self.field_combo.currentText()}")
            self.log(f"Min frequency: {self.min_spin.value() if self.min_spin.value() > 0 else 'None'}")
            self.log(f"Max frequency: {self.max_spin.value() if self.max_spin.value() > 0 else 'None'}")
            self.log("")
            
            # Disable controls
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.input_browse.setEnabled(False)
            self.output_browse.setEnabled(False)
            
            # Start worker
            self.worker = FilterWorker(
                input_path,
                output_path,
                self.field_combo.currentText(),
                self.min_spin.value() if self.min_spin.value() > 0 else None,
                self.max_spin.value() if self.max_spin.value() > 0 else None,
            )
            
            self.worker.progress_update.connect(self.log)
            self.worker.finished.connect(self.on_finished)
            self.worker.start()
        
        def stop_filtering(self):
            """Stop the filtering process."""
            if self.worker:
                self.log("Stopping...")
                self.worker.request_stop()
                self.worker.wait()
                self.on_finished(False, "Filtering stopped by user")
        
        def on_finished(self, success: bool, message: str):
            """Handle filtering completion."""
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
    window = FilterSpaCyVerbsGUI()
    window.show()
    sys.exit(app.exec())


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point - choose GUI or CLI mode."""
    if len(sys.argv) == 1:
        print("Launching GUI mode...")
        print("(Use command-line arguments to run in CLI mode. Use --help for usage.)")
        run_gui()
    else:
        run_cli()


if __name__ == "__main__":
    main()