#!/usr/bin/env python3
"""
Spell check text files or CSVs with CLI + GUI + metadata support.

Supports both CLI and GUI modes:
  - Run without arguments: launches GUI (requires PySide6)
  - Run with arguments: uses CLI mode

Two spell checking modes:
  - Auto: Automatically replaces misspellings with the most likely correction
  - Interactive: Prompts user for each misspelling (replace, ignore once, ignore all, add to dictionary)

Features:
  - Custom dictionary support (words to add to spell checker)
  - Ignore list support (words to ignore but not use as corrections)
  - Metadata tracking with checksums
  - CSV and text file support

Requirements:
  pip install pyenchant PySide6

CLI Examples:
  # Auto mode - replace with top suggestion
  python SpellChecker.py input.txt output.txt --mode auto
  
  # Interactive mode - prompt for each misspelling
  python SpellChecker.py input.txt output.txt --mode interactive
  
  # With custom dictionary and ignore list
  python SpellChecker.py input.csv output.csv --mode auto --custom-dict mywords.txt --ignore-list ignore.txt --text-column text
  
  # Load from metadata
  python SpellChecker.py --load-metadata output.json input.txt output2.txt

GUI Mode:
  python SpellChecker.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

try:
    import enchant
except ImportError:
    print("Error: pyenchant is not installed. Install with: pip install pyenchant", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# VERSION TRACKING
# ============================================================================

def get_spell_checker_version_info() -> Dict:
    """Get pyenchant version information."""
    import enchant
    return {
        "pyenchant": enchant.__version__ if hasattr(enchant, '__version__') else "unknown",
    }


def check_spell_checker_version_compatibility(metadata: Dict) -> str:
    """Check if pyenchant version matches metadata and return warning if different."""
    if "versions" not in metadata:
        return ""
    
    meta_versions = metadata.get("versions", {})
    current_versions = get_spell_checker_version_info()
    
    if meta_versions.get("pyenchant") != current_versions.get("pyenchant"):
        return (
            f"⚠ pyenchant version mismatch: metadata has {meta_versions.get('pyenchant')}, "
            f"current is {current_versions.get('pyenchant')}"
        )
    return ""


def reconstruct_command(input_file: str, output_file: str, args: argparse.Namespace) -> str:
    """Reconstruct the CLI command that would reproduce this run."""
    cmd = ["python", "SpellChecker.py", input_file, output_file]
    
    if args.mode != "auto":
        cmd.extend(["--mode", args.mode])
    if args.language != "en_US":
        cmd.extend(["--language", args.language])
    if args.custom_dict:
        cmd.extend(["--custom-dict", args.custom_dict])
    if args.ignore_list:
        cmd.extend(["--ignore-list", args.ignore_list])
    if args.text_column:
        cmd.extend(["--text-column", args.text_column])
    if args.csv_format != "complete":
        cmd.extend(["--csv-format", args.csv_format])
    if args.text_format != "corrected":
        cmd.extend(["--text-format", args.text_format])
    if args.encoding != "utf-8":
        cmd.extend(["--encoding", args.encoding])
    
    return " ".join(cmd)
def compute_file_md5(path: Path) -> str:
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def save_spell_metadata(
    output_path: Path,
    input_path: Path,
    input_checksum: str,
    output_checksum: str,
    args: Dict,
    stats: Dict,
    custom_dict_path: Optional[Path] = None,
    ignore_list_path: Optional[Path] = None,
    source_metadata: Optional[Dict] = None,
    command: Optional[str] = None,
) -> None:
    """Save spell check metadata to JSON file alongside output."""
    json_path = output_path.with_suffix(".json")
    
    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tool": "SpellChecker",
        "versions": get_spell_checker_version_info(),
        "input_file": str(input_path),
        "input_checksum": input_checksum,
        "output_file": str(output_path),
        "output_checksum": output_checksum,
        "command": command,
        "settings": {
            "mode": args.get("mode", "auto"),
            "language": args.get("language", "en_US"),
            "custom_dict": str(custom_dict_path) if custom_dict_path else None,
            "ignore_list": str(ignore_list_path) if ignore_list_path else None,
            "text_column": args.get("text_column"),
            "csv_format": args.get("csv_format"),
            "text_format": args.get("text_format"),
            "encoding": args.get("encoding", "utf-8"),
        },
        "statistics": stats,
    }
    
    # Include source metadata for pipeline chaining
    if source_metadata:
        metadata["source_metadata"] = source_metadata
    
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_spell_metadata(json_path: Path) -> Dict:
    """Load spell check metadata from JSON file."""
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
# CUSTOM DICTIONARY AND IGNORE LIST FUNCTIONS
# ============================================================================

def load_word_list(path: Optional[Path]) -> Set[str]:
    """Load a word list from a text file (one word per line)."""
    if not path or not path.exists():
        return set()
    
    words = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word and not word.startswith("#"):
                words.add(word.lower())
    return words


def save_word_list(path: Path, words: Set[str]) -> None:
    """Save a word list to a text file (one word per line, sorted)."""
    with path.open("w", encoding="utf-8") as f:
        for word in sorted(words):
            f.write(f"{word}\n")


# ============================================================================
# CORE SPELL CHECKING FUNCTIONS
# ============================================================================

class SpellCheckerEngine:
    """Spell checker engine with custom dictionary and ignore list support."""
    
    def __init__(
        self,
        language: str = "en_US",
        custom_dict_path: Optional[Path] = None,
        ignore_list_path: Optional[Path] = None,
    ):
        self.dict = enchant.Dict(language)
        self.custom_words = load_word_list(custom_dict_path)
        self.ignore_words = load_word_list(ignore_list_path)
        self.custom_dict_path = custom_dict_path
        self.ignore_list_path = ignore_list_path
        
        # Add custom words to the dictionary
        for word in self.custom_words:
            if word:
                self.dict.add(word)
    
    def is_correct(self, word: str) -> bool:
        """Check if a word is spelled correctly or should be ignored."""
        word_lower = word.lower()
        if word_lower in self.ignore_words:
            return True
        return self.dict.check(word)
    
    def suggest(self, word: str, max_suggestions: int = 10) -> List[str]:
        """Get spelling suggestions for a misspelled word."""
        suggestions = self.dict.suggest(word)
        # Filter out ignored words from suggestions
        filtered = [s for s in suggestions if s.lower() not in self.ignore_words]
        return filtered[:max_suggestions]
    
    def add_to_custom_dict(self, word: str) -> None:
        """Add a word to the custom dictionary."""
        word_lower = word.lower()
        self.custom_words.add(word_lower)
        self.dict.add(word_lower)
        if self.custom_dict_path:
            save_word_list(self.custom_dict_path, self.custom_words)
    
    def add_to_ignore_list(self, word: str) -> None:
        """Add a word to the ignore list."""
        word_lower = word.lower()
        self.ignore_words.add(word_lower)
        if self.ignore_list_path:
            save_word_list(self.ignore_list_path, self.ignore_words)


def check_text_auto(
    text: str,
    checker: SpellCheckerEngine,
) -> Tuple[str, int, int]:
    """
    Auto mode: Replace misspellings with top suggestion.
    
    Returns (corrected_text, corrections_made, words_checked)
    """
    words = text.split()
    corrections_made = 0
    words_checked = 0
    
    corrected_words = []
    for word in words:
        # Strip punctuation for checking but preserve it
        stripped = word.strip(".,!?;:\"'()[]{}").strip()
        if not stripped:
            corrected_words.append(word)
            continue
        
        words_checked += 1
        
        if not checker.is_correct(stripped):
            suggestions = checker.suggest(stripped, max_suggestions=1)
            if suggestions:
                # Replace the word, preserving punctuation
                replacement = suggestions[0]
                corrected_word = word.replace(stripped, replacement)
                corrected_words.append(corrected_word)
                corrections_made += 1
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    
    return " ".join(corrected_words), corrections_made, words_checked


def check_text_interactive(
    text: str,
    checker: SpellCheckerEngine,
    callback,
) -> Tuple[str, int, int]:
    """
    Interactive mode: Prompt user for each misspelling.
    
    callback(word, suggestions) should return:
      - ("replace", replacement_word)
      - ("ignore_once",)
      - ("ignore_all",)
      - ("add_to_dict",)
    
    Returns (corrected_text, corrections_made, words_checked)
    """
    words = text.split()
    corrections_made = 0
    words_checked = 0
    
    corrected_words = []
    for word in words:
        # Strip punctuation for checking but preserve it
        stripped = word.strip(".,!?;:\"'()[]{}").strip()
        if not stripped:
            corrected_words.append(word)
            continue
        
        words_checked += 1
        
        if not checker.is_correct(stripped):
            suggestions = checker.suggest(stripped)
            action = callback(stripped, suggestions)
            
            if action[0] == "replace":
                replacement = action[1]
                corrected_word = word.replace(stripped, replacement)
                corrected_words.append(corrected_word)
                corrections_made += 1
            elif action[0] == "ignore_once":
                corrected_words.append(word)
            elif action[0] == "ignore_all":
                checker.add_to_ignore_list(stripped)
                corrected_words.append(word)
            elif action[0] == "add_to_dict":
                checker.add_to_custom_dict(stripped)
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    
    return " ".join(corrected_words), corrections_made, words_checked


# ============================================================================
# CLI MODE
# ============================================================================

def cli_interactive_callback(word: str, suggestions: List[str]) -> Tuple[str, ...]:
    """CLI callback for interactive mode."""
    print(f"\nMisspelled word: '{word}'")
    print("Suggestions:")
    for i, sugg in enumerate(suggestions[:10], 1):
        print(f"  {i}. {sugg}")
    
    print("\nOptions:")
    print("  [1-9] - Replace with suggestion number")
    print("  [i] - Ignore once")
    print("  [a] - Ignore all (add to ignore list)")
    print("  [d] - Add to custom dictionary")
    print("  [Enter] - Ignore once")
    
    choice = input("Your choice: ").strip().lower()
    
    if choice == "":
        return ("ignore_once",)
    elif choice == "i":
        return ("ignore_once",)
    elif choice == "a":
        return ("ignore_all",)
    elif choice == "d":
        return ("add_to_dict",)
    elif choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(suggestions):
            return ("replace", suggestions[idx])
        else:
            print("Invalid choice, ignoring once")
            return ("ignore_once",)
    else:
        print("Invalid choice, ignoring once")
        return ("ignore_once",)


def run_cli() -> None:
    """Run CLI mode."""
    ap = argparse.ArgumentParser(
        description="Spell check text files or CSVs with custom dictionaries and ignore lists"
    )
    ap.add_argument("input_file", nargs="?", help="Input file (text or CSV)")
    ap.add_argument("output_file", nargs="?", help="Output file")
    ap.add_argument("--mode", choices=["auto", "interactive"], default="auto", help="Spell check mode (default: auto)")
    ap.add_argument("--language", default="en_US", help="Spell check language (default: en_US)")
    ap.add_argument("--custom-dict", help="Path to custom dictionary file (one word per line)")
    ap.add_argument("--ignore-list", help="Path to ignore list file (one word per line)")
    ap.add_argument("--text-column", help="CSV column name containing text to check")
    ap.add_argument("--csv-format", choices=["complete", "two-column", "single-column"], default="complete",
                    help="CSV output format: complete (full CSV + corrected column), two-column (original + corrected), single-column (corrected only) (default: complete)")
    ap.add_argument("--text-format", choices=["corrected", "patch"], default="corrected",
                    help="Text output format: corrected (corrected text), patch (unified diff patch) (default: corrected)")
    ap.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    ap.add_argument("--load-metadata", help="Load settings from metadata JSON (CLI args override)")
    ap.add_argument("--strict-checksum", action="store_true", help="Abort if input file checksum mismatches metadata")
    args = ap.parse_args()
    
    # Load metadata if provided
    metadata = None
    source_metadata = None
    if args.load_metadata:
        metadata_path = Path(args.load_metadata)
        if not metadata_path.exists():
            raise SystemExit(f"Metadata file not found: {metadata_path}")
        
        metadata = load_spell_metadata(metadata_path)
        tool_name = metadata.get("tool", "unknown")
        
        # Check version compatibility
        version_warning = check_spell_checker_version_compatibility(metadata)
        if version_warning:
            print(version_warning)
        
        # Handle SpellChecker metadata
        if tool_name == "SpellChecker":
            source_metadata = metadata.get("source_metadata")
            if not args.input_file:
                args.input_file = metadata.get("input_file")
        # Handle SpaCyVerbExtractor, FilterSpaCyVerbs, or RoBERTaMaskedLanguageModelVerbs
        elif tool_name in ["SpaCyVerbExtractor", "FilterSpaCyVerbs", "RoBERTaMaskedLanguageModelVerbs"]:
            source_metadata = metadata
            if not args.input_file:
                args.input_file = metadata.get("output_file")
        else:
            source_metadata = metadata.get("source_metadata")
            if not args.input_file:
                args.input_file = metadata.get("input_file")
        
        # Use loaded settings as defaults
        settings = metadata.get("settings", {})
        if args.mode == "auto":
            args.mode = settings.get("mode", "auto")
        if args.language == "en_US":
            args.language = settings.get("language", "en_US")
        if not args.custom_dict:
            args.custom_dict = settings.get("custom_dict")
        if not args.ignore_list:
            args.ignore_list = settings.get("ignore_list")
        if not args.text_column:
            args.text_column = settings.get("text_column")
        if args.csv_format == "complete":
            args.csv_format = settings.get("csv_format", "complete")
        if args.text_format == "corrected":
            args.text_format = settings.get("text_format", "corrected")
    
    # Validate required arguments
    if not args.input_file or not args.output_file:
        ap.print_help()
        raise SystemExit("Error: input_file and output_file are required (or use --load-metadata)")
    
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    
    # Verify checksum if metadata was loaded
    if metadata:
        matches, message = verify_input_checksum(input_path, metadata)
        if not matches:
            if args.strict_checksum:
                raise SystemExit(f"Checksum mismatch: {message}")
            else:
                print(f"⚠ {message}")
    
    # Set up custom dictionary and ignore list paths
    custom_dict_path = Path(args.custom_dict) if args.custom_dict else None
    ignore_list_path = Path(args.ignore_list) if args.ignore_list else None
    
    # Initialize spell checker
    print(f"Initializing spell checker (language: {args.language})...")
    checker = SpellCheckerEngine(args.language, custom_dict_path, ignore_list_path)
    
    # Process file
    is_csv = input_path.suffix.lower() == ".csv"
    total_corrections = 0
    total_words_checked = 0
    total_rows = 0
    
    print(f"Processing {input_path}...")
    
    if is_csv:
        if not args.text_column:
            raise SystemExit("Error: --text-column is required for CSV files")
        
        with input_path.open("r", newline="", encoding=args.encoding) as fin, \
             output_path.open("w", newline="", encoding=args.encoding) as fout:
            
            reader = csv.DictReader(fin)
            if not reader.fieldnames:
                raise SystemExit("Error: CSV has no headers")
            
            if args.text_column not in reader.fieldnames:
                raise SystemExit(f"Error: Column '{args.text_column}' not found in CSV")
            
            # Determine output fieldnames based on format
            if args.csv_format == "complete":
                # Full CSV + corrected column
                out_fieldnames = list(reader.fieldnames) + [args.text_column + "_corrected"]
            elif args.csv_format == "two-column":
                # Original + corrected
                out_fieldnames = [args.text_column, args.text_column + "_corrected"]
            else:  # single-column
                # Corrected only
                out_fieldnames = [args.text_column + "_corrected"]
            
            writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
            writer.writeheader()
            
            for row in reader:
                total_rows += 1
                text = row[args.text_column]
                
                if args.mode == "auto":
                    corrected, corrections, words_checked = check_text_auto(text, checker)
                else:
                    corrected, corrections, words_checked = check_text_interactive(
                        text, checker, cli_interactive_callback
                    )
                
                # Prepare output row based on format
                if args.csv_format == "complete":
                    # Keep all columns + add corrected
                    out_row = row.copy()
                    out_row[args.text_column + "_corrected"] = corrected
                elif args.csv_format == "two-column":
                    # Original + corrected
                    out_row = {
                        args.text_column: text,
                        args.text_column + "_corrected": corrected
                    }
                else:  # single-column
                    # Corrected only
                    out_row = {args.text_column + "_corrected": corrected}
                
                writer.writerow(out_row)
                
                total_corrections += corrections
                total_words_checked += words_checked
                
                if total_rows % 100 == 0:
                    print(f"Processed {total_rows} rows ({total_corrections} corrections, {total_words_checked} words checked)")
    else:
        # Text file
        if args.text_format == "patch":
            # Generate patch file
            import difflib
            
            original_lines = []
            corrected_lines = []
            
            with input_path.open("r", encoding=args.encoding) as fin:
                for line_num, line in enumerate(fin, 1):
                    total_rows += 1
                    line_content = line.rstrip("\n")
                    original_lines.append(line)
                    
                    if args.mode == "auto":
                        corrected, corrections, words_checked = check_text_auto(line_content, checker)
                    else:
                        corrected, corrections, words_checked = check_text_interactive(
                            line_content, checker, cli_interactive_callback
                        )
                    
                    corrected_lines.append(corrected + "\n")
                    
                    total_corrections += corrections
                    total_words_checked += words_checked
                    
                    if line_num % 100 == 0:
                        print(f"Processed {line_num} lines ({total_corrections} corrections, {total_words_checked} words checked)")
            
            # Write unified diff
            diff = difflib.unified_diff(
                original_lines,
                corrected_lines,
                fromfile=str(input_path),
                tofile=str(output_path),
                lineterm=''
            )
            
            with output_path.open("w", encoding=args.encoding) as fout:
                for line in diff:
                    fout.write(line + "\n")
        else:
            # Corrected text output
            with input_path.open("r", encoding=args.encoding) as fin, \
                 output_path.open("w", encoding=args.encoding) as fout:
                
                for line_num, line in enumerate(fin, 1):
                    total_rows += 1
                    
                    if args.mode == "auto":
                        corrected, corrections, words_checked = check_text_auto(line.rstrip("\n"), checker)
                    else:
                        corrected, corrections, words_checked = check_text_interactive(
                            line.rstrip("\n"), checker, cli_interactive_callback
                        )
                    
                    fout.write(corrected + "\n")
                    
                    total_corrections += corrections
                    total_words_checked += words_checked
                    
                    if line_num % 100 == 0:
                        print(f"Processed {line_num} lines ({total_corrections} corrections, {total_words_checked} words checked)")
    
    # Compute checksums and save metadata
    input_checksum = compute_file_md5(input_path)
    output_checksum = compute_file_md5(output_path)
    
    spell_args = {
        "mode": args.mode,
        "language": args.language,
        "custom_dict": args.custom_dict,
        "ignore_list": args.ignore_list,
        "text_column": args.text_column,
        "csv_format": args.csv_format if is_csv else None,
        "text_format": args.text_format if not is_csv else None,
        "encoding": args.encoding,
    }
    
    stats = {
        "total_rows": total_rows,
        "total_corrections": total_corrections,
        "total_words_checked": total_words_checked,
    }
    
    command = reconstruct_command(args.input_file, args.output_file, args)
    
    save_spell_metadata(
        output_path, input_path, input_checksum, output_checksum,
        spell_args, stats, custom_dict_path, ignore_list_path, source_metadata, command
    )
    
    print(f"\n✓ Processed {total_rows} {'rows' if is_csv else 'lines'}")
    print(f"✓ Made {total_corrections} corrections ({total_words_checked} words checked)")
    print(f"✓ Saved to {output_path}")
    print(f"✓ Saved metadata to {output_path.with_suffix('.json')}")


# ============================================================================
# GUI MODE
# ============================================================================

def run_gui() -> None:
    """Run GUI mode."""
    try:
        from PySide6.QtCore import Qt, QThread, Signal, QSize
        from PySide6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog,
            QMessageBox, QGroupBox, QComboBox, QCheckBox, QDialog,
            QDialogButtonBox, QListWidget
        )
    except ImportError:
        print("Error: PySide6 is not installed. Install with: pip install PySide6", file=sys.stderr)
        sys.exit(1)
    
    class CorrectionDialog(QDialog):
        """Dialog for interactive spell correction."""
        
        def __init__(self, word: str, suggestions: List[str], parent=None):
            super().__init__(parent)
            self.setWindowTitle(f"Spell Check: {word}")
            self.setMinimumWidth(400)
            
            self.word = word
            self.result = None
            
            layout = QVBoxLayout(self)
            
            # Word label
            label = QLabel(f"Misspelled word: <b>{word}</b>")
            layout.addWidget(label)
            
            # Suggestions list
            sugg_label = QLabel("Suggestions:")
            layout.addWidget(sugg_label)
            
            self.sugg_list = QListWidget()
            self.sugg_list.addItems(suggestions[:20])
            if suggestions:
                self.sugg_list.setCurrentRow(0)
            self.sugg_list.itemDoubleClicked.connect(self.replace_clicked)
            layout.addWidget(self.sugg_list)
            
            # Buttons
            button_layout = QVBoxLayout()
            
            replace_btn = QPushButton("Replace")
            replace_btn.clicked.connect(self.replace_clicked)
            button_layout.addWidget(replace_btn)
            
            ignore_once_btn = QPushButton("Ignore Once")
            ignore_once_btn.clicked.connect(self.ignore_once_clicked)
            button_layout.addWidget(ignore_once_btn)
            
            ignore_all_btn = QPushButton("Ignore All")
            ignore_all_btn.clicked.connect(self.ignore_all_clicked)
            button_layout.addWidget(ignore_all_btn)
            
            add_dict_btn = QPushButton("Add to Dictionary")
            add_dict_btn.clicked.connect(self.add_dict_clicked)
            button_layout.addWidget(add_dict_btn)
            
            layout.addLayout(button_layout)
        
        def replace_clicked(self):
            """Replace with selected suggestion."""
            current = self.sugg_list.currentItem()
            if current:
                self.result = ("replace", current.text())
                self.accept()
        
        def ignore_once_clicked(self):
            """Ignore this occurrence."""
            self.result = ("ignore_once",)
            self.accept()
        
        def ignore_all_clicked(self):
            """Ignore all occurrences."""
            self.result = ("ignore_all",)
            self.accept()
        
        def add_dict_clicked(self):
            """Add to custom dictionary."""
            self.result = ("add_to_dict",)
            self.accept()
    
    class SpellCheckWorker(QThread):
        """Worker thread for spell checking."""
        
        progress_update = Signal(str)
        correction_needed = Signal(str, list)  # word, suggestions
        finished = Signal(bool, str)
        
        def __init__(
            self,
            input_path: Path,
            output_path: Path,
            mode: str,
            language: str,
            custom_dict_path: Optional[Path],
            ignore_list_path: Optional[Path],
            text_column: Optional[str],
            encoding: str,
            source_metadata: Optional[Dict],
        ):
            super().__init__()
            self.input_path = input_path
            self.output_path = output_path
            self.mode = mode
            self.language = language
            self.custom_dict_path = custom_dict_path
            self.ignore_list_path = ignore_list_path
            self.text_column = text_column
            self.encoding = encoding
            self.source_metadata = source_metadata
            self.csv_format = "complete"
            self.text_format = "corrected"
            self._stop_requested = False
            self._correction_result = None
        
        def request_stop(self):
            """Request graceful stop."""
            self._stop_requested = True
        
        def set_correction_result(self, result):
            """Set the correction result from interactive dialog."""
            self._correction_result = result
        
        def interactive_callback(self, word: str, suggestions: List[str]) -> Tuple[str, ...]:
            """Callback for interactive mode - emit signal and wait."""
            self._correction_result = None
            self.correction_needed.emit(word, suggestions)
            
            # Wait for result (busy wait with sleep)
            while self._correction_result is None and not self._stop_requested:
                self.msleep(50)
            
            if self._stop_requested:
                return ("ignore_once",)
            
            return self._correction_result
        
        def run(self):
            """Run spell checking in worker thread."""
            try:
                self.progress_update.emit(f"Initializing spell checker (language: {self.language})...")
                checker = SpellCheckerEngine(
                    self.language, self.custom_dict_path, self.ignore_list_path
                )
                
                is_csv = self.input_path.suffix.lower() == ".csv"
                total_corrections = 0
                total_words_checked = 0
                total_rows = 0
                
                self.progress_update.emit(f"Processing {self.input_path.name}...")
                
                if is_csv:
                    if not self.text_column:
                        raise ValueError("Text column is required for CSV files")
                    
                    with self.input_path.open("r", newline="", encoding=self.encoding) as fin, \
                         self.output_path.open("w", newline="", encoding=self.encoding) as fout:
                        
                        reader = csv.DictReader(fin)
                        if not reader.fieldnames:
                            raise ValueError("CSV has no headers")
                        
                        if self.text_column not in reader.fieldnames:
                            raise ValueError(f"Column '{self.text_column}' not found in CSV")
                        
                        # Determine output fieldnames based on format
                        if self.csv_format == "complete":
                            out_fieldnames = list(reader.fieldnames) + [self.text_column + "_corrected"]
                        elif self.csv_format == "two-column":
                            out_fieldnames = [self.text_column, self.text_column + "_corrected"]
                        else:  # single-column
                            out_fieldnames = [self.text_column + "_corrected"]
                        
                        writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
                        writer.writeheader()
                        
                        for row in reader:
                            if self._stop_requested:
                                break
                            
                            total_rows += 1
                            text = row[self.text_column]
                            
                            if self.mode == "auto":
                                corrected, corrections, words_checked = check_text_auto(text, checker)
                            else:
                                corrected, corrections, words_checked = check_text_interactive(
                                    text, checker, self.interactive_callback
                                )
                            
                            # Prepare output row based on format
                            if self.csv_format == "complete":
                                out_row = row.copy()
                                out_row[self.text_column + "_corrected"] = corrected
                            elif self.csv_format == "two-column":
                                out_row = {
                                    self.text_column: text,
                                    self.text_column + "_corrected": corrected
                                }
                            else:  # single-column
                                out_row = {self.text_column + "_corrected": corrected}
                            
                            writer.writerow(out_row)
                            
                            total_corrections += corrections
                            total_words_checked += words_checked
                            
                            if total_rows % 50 == 0:
                                self.progress_update.emit(
                                    f"Processed {total_rows} rows ({total_corrections} corrections, {total_words_checked} words checked)"
                                )
                else:
                    # Text file
                    if self.text_format == "patch":
                        # Generate patch file
                        import difflib
                        
                        original_lines = []
                        corrected_lines = []
                        
                        with self.input_path.open("r", encoding=self.encoding) as fin:
                            for line_num, line in enumerate(fin, 1):
                                if self._stop_requested:
                                    break
                                
                                total_rows += 1
                                line_content = line.rstrip("\n")
                                original_lines.append(line)
                                
                                if self.mode == "auto":
                                    corrected, corrections, words_checked = check_text_auto(line_content, checker)
                                else:
                                    corrected, corrections, words_checked = check_text_interactive(
                                        line_content, checker, self.interactive_callback
                                    )
                                
                                corrected_lines.append(corrected + "\n")
                                
                                total_corrections += corrections
                                total_words_checked += words_checked
                                
                                if line_num % 50 == 0:
                                    self.progress_update.emit(
                                        f"Processed {line_num} lines ({total_corrections} corrections, {total_words_checked} words checked)"
                                    )
                        
                        if not self._stop_requested:
                            # Write unified diff
                            diff = difflib.unified_diff(
                                original_lines,
                                corrected_lines,
                                fromfile=str(self.input_path),
                                tofile=str(self.output_path),
                                lineterm=''
                            )
                            
                            with self.output_path.open("w", encoding=self.encoding) as fout:
                                for line in diff:
                                    fout.write(line + "\n")
                    else:
                        # Corrected text output
                        with self.input_path.open("r", encoding=self.encoding) as fin, \
                             self.output_path.open("w", encoding=self.encoding) as fout:
                            
                            for line_num, line in enumerate(fin, 1):
                                if self._stop_requested:
                                    break
                                
                                total_rows += 1
                                
                                if self.mode == "auto":
                                    corrected, corrections, words_checked = check_text_auto(line.rstrip("\n"), checker)
                                else:
                                    corrected, corrections, words_checked = check_text_interactive(
                                        line.rstrip("\n"), checker, self.interactive_callback
                                    )
                                
                                fout.write(corrected + "\n")
                                
                                total_corrections += corrections
                                total_words_checked += words_checked
                                
                                if line_num % 50 == 0:
                                    self.progress_update.emit(
                                        f"Processed {line_num} lines ({total_corrections} corrections, {total_words_checked} words checked)"
                                    )
                
                if self._stop_requested:
                    self.finished.emit(False, "Spell check stopped by user")
                    return
                
                # Save metadata
                input_checksum = compute_file_md5(self.input_path)
                output_checksum = compute_file_md5(self.output_path)
                
                spell_args = {
                    "mode": self.mode,
                    "language": self.language,
                    "custom_dict": str(self.custom_dict_path) if self.custom_dict_path else None,
                    "ignore_list": str(self.ignore_list_path) if self.ignore_list_path else None,
                    "text_column": self.text_column,
                    "csv_format": self.csv_format if is_csv else None,
                    "text_format": self.text_format if not is_csv else None,
                    "encoding": self.encoding,
                }
                
                stats = {
                    "total_rows": total_rows,
                    "total_corrections": total_corrections,
                    "total_words_checked": total_words_checked,
                }
                
                # Reconstruct command for GUI mode
                gui_command = (
                    f"python SpellChecker.py {self.input_path} {self.output_path} "
                    f"--mode {self.mode} --language {self.language}"
                )
                if self.custom_dict_path:
                    gui_command += f" --custom-dict {self.custom_dict_path}"
                if self.ignore_list_path:
                    gui_command += f" --ignore-list {self.ignore_list_path}"
                if text_column:
                    gui_command += f" --text-column {text_column}"
                if self.csv_format != "complete":
                    gui_command += f" --csv-format {self.csv_format}"
                if self.text_format != "corrected":
                    gui_command += f" --text-format {self.text_format}"
                
                save_spell_metadata(
                    self.output_path, self.input_path, input_checksum, output_checksum,
                    spell_args, stats, self.custom_dict_path, self.ignore_list_path,
                    self.source_metadata, gui_command
                )
                
                self.progress_update.emit(f"✓ Processed {total_rows} {'rows' if is_csv else 'lines'}")
                self.progress_update.emit(f"✓ Made {total_corrections} corrections ({total_words_checked} words checked)")
                self.progress_update.emit(f"✓ Saved metadata: {self.output_path.with_suffix('.json')}")
                
                msg = f"Spell check complete.\n{total_corrections} corrections made ({total_words_checked} words checked)."
                self.finished.emit(True, msg)
                
            except Exception as e:
                self.progress_update.emit(f"✗ Error: {str(e)}")
                self.finished.emit(False, f"Error: {str(e)}")
    
    class SpellCheckerGUI(QMainWindow):
        """Main GUI window for spell checker."""
        
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Spell Checker")
            self.setMinimumSize(QSize(900, 700))
            
            self.worker: Optional[SpellCheckWorker] = None
            self.source_metadata: Optional[Dict] = None
            
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
            input_label = QLabel("Input File:")
            self.input_text = QLineEdit()
            self.input_browse = QPushButton("Browse...")
            self.input_browse.clicked.connect(self.browse_input)
            input_layout.addWidget(input_label)
            input_layout.addWidget(self.input_text)
            input_layout.addWidget(self.input_browse)
            files_layout.addLayout(input_layout)
            
            # Output file
            output_layout = QHBoxLayout()
            output_label = QLabel("Output File:")
            self.output_text = QLineEdit()
            self.output_browse = QPushButton("Browse...")
            self.output_browse.clicked.connect(self.browse_output)
            output_layout.addWidget(output_label)
            output_layout.addWidget(self.output_text)
            output_layout.addWidget(self.output_browse)
            files_layout.addLayout(output_layout)
            
            # Load metadata
            metadata_layout = QHBoxLayout()
            self.load_metadata_btn = QPushButton("Load Settings from JSON")
            self.load_metadata_btn.clicked.connect(self.load_metadata_dialog)
            metadata_layout.addStretch()
            metadata_layout.addWidget(self.load_metadata_btn)
            files_layout.addLayout(metadata_layout)
            
            files_group.setLayout(files_layout)
            layout.addWidget(files_group)
            
            # Settings section
            settings_group = QGroupBox("Settings")
            settings_layout = QVBoxLayout()
            
            # Mode
            mode_layout = QHBoxLayout()
            mode_label = QLabel("Mode:")
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["auto", "interactive"])
            mode_layout.addWidget(mode_label)
            mode_layout.addWidget(self.mode_combo)
            mode_layout.addStretch()
            settings_layout.addLayout(mode_layout)
            
            # Language
            lang_layout = QHBoxLayout()
            lang_label = QLabel("Language:")
            self.lang_text = QLineEdit("en_US")
            lang_layout.addWidget(lang_label)
            lang_layout.addWidget(self.lang_text)
            lang_layout.addStretch()
            settings_layout.addLayout(lang_layout)
            
            # Custom dictionary
            custom_dict_layout = QHBoxLayout()
            custom_dict_label = QLabel("Custom Dict:")
            self.custom_dict_text = QLineEdit()
            self.custom_dict_browse = QPushButton("Browse...")
            self.custom_dict_browse.clicked.connect(self.browse_custom_dict)
            custom_dict_layout.addWidget(custom_dict_label)
            custom_dict_layout.addWidget(self.custom_dict_text)
            custom_dict_layout.addWidget(self.custom_dict_browse)
            settings_layout.addLayout(custom_dict_layout)
            
            # Ignore list
            ignore_list_layout = QHBoxLayout()
            ignore_list_label = QLabel("Ignore List:")
            self.ignore_list_text = QLineEdit()
            self.ignore_list_browse = QPushButton("Browse...")
            self.ignore_list_browse.clicked.connect(self.browse_ignore_list)
            ignore_list_layout.addWidget(ignore_list_label)
            ignore_list_layout.addWidget(self.ignore_list_text)
            ignore_list_layout.addWidget(self.ignore_list_browse)
            settings_layout.addLayout(ignore_list_layout)
            
            # CSV text column
            text_col_layout = QHBoxLayout()
            text_col_label = QLabel("CSV Text Column:")
            self.text_col_text = QLineEdit()
            self.text_col_text.setPlaceholderText("(for CSV files only)")
            text_col_layout.addWidget(text_col_label)
            text_col_layout.addWidget(self.text_col_text)
            text_col_layout.addStretch()
            settings_layout.addLayout(text_col_layout)
            
            # CSV output format
            csv_fmt_layout = QHBoxLayout()
            csv_fmt_label = QLabel("CSV Output Format:")
            self.csv_format_combo = QComboBox()
            self.csv_format_combo.addItems(["complete", "two-column", "single-column"])
            self.csv_format_combo.setToolTip(
                "complete: Full CSV + corrected column\n"
                "two-column: Original + corrected columns only\n"
                "single-column: Corrected text only"
            )
            csv_fmt_layout.addWidget(csv_fmt_label)
            csv_fmt_layout.addWidget(self.csv_format_combo)
            csv_fmt_layout.addStretch()
            settings_layout.addLayout(csv_fmt_layout)
            
            # Text output format
            text_fmt_layout = QHBoxLayout()
            text_fmt_label = QLabel("Text Output Format:")
            self.text_format_combo = QComboBox()
            self.text_format_combo.addItems(["corrected", "patch"])
            self.text_format_combo.setToolTip(
                "corrected: Corrected text file\n"
                "patch: Unified diff patch file"
            )
            text_fmt_layout.addWidget(text_fmt_label)
            text_fmt_layout.addWidget(self.text_format_combo)
            text_fmt_layout.addStretch()
            settings_layout.addLayout(text_fmt_layout)
            
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
            self.start_btn = QPushButton("Start Spell Check")
            self.start_btn.clicked.connect(self.start_spell_check)
            self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
            self.stop_btn = QPushButton("Stop")
            self.stop_btn.clicked.connect(self.stop_spell_check)
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
                "Select Input File",
                "",
                "All Files (*.txt *.csv);;Text Files (*.txt);;CSV Files (*.csv)"
            )
            if file:
                self.input_text.setText(file)
        
        def browse_output(self):
            """Open file dialog for output file."""
            file, _ = QFileDialog.getSaveFileName(
                self,
                "Select Output File",
                "corrected.txt",
                "All Files (*.txt *.csv);;Text Files (*.txt);;CSV Files (*.csv)"
            )
            if file:
                self.output_text.setText(file)
        
        def browse_custom_dict(self):
            """Open file dialog for custom dictionary."""
            file, _ = QFileDialog.getOpenFileName(
                self,
                "Select Custom Dictionary",
                "",
                "Text Files (*.txt);;All Files (*)"
            )
            if file:
                self.custom_dict_text.setText(file)
        
        def browse_ignore_list(self):
            """Open file dialog for ignore list."""
            file, _ = QFileDialog.getOpenFileName(
                self,
                "Select Ignore List",
                "",
                "Text Files (*.txt);;All Files (*)"
            )
            if file:
                self.ignore_list_text.setText(file)
        
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
                metadata = load_spell_metadata(json_path)
                tool = metadata.get("tool", "unknown")
                settings = metadata.get("settings", {})
                
                if tool == "SpellChecker":
                    self.input_text.setText(metadata.get("input_file", ""))
                    self.output_text.setText(metadata.get("output_file", ""))
                    self.mode_combo.setCurrentText(settings.get("mode", "auto"))
                    self.lang_text.setText(settings.get("language", "en_US"))
                    self.custom_dict_text.setText(settings.get("custom_dict", ""))
                    self.ignore_list_text.setText(settings.get("ignore_list", ""))
                    self.text_col_text.setText(settings.get("text_column", ""))
                    
                    csv_fmt = settings.get("csv_format", "complete")
                    idx = self.csv_format_combo.findText(csv_fmt)
                    if idx >= 0:
                        self.csv_format_combo.setCurrentIndex(idx)
                    
                    text_fmt = settings.get("text_format", "corrected")
                    idx = self.text_format_combo.findText(text_fmt)
                    if idx >= 0:
                        self.text_format_combo.setCurrentIndex(idx)
                    
                    self.source_metadata = metadata.get("source_metadata")
                    self.log(f"✓ Loaded settings from: {json_path}")
                elif tool in ["SpaCyVerbExtractor", "FilterSpaCyVerbs", "RoBERTaMaskedLanguageModelVerbs"]:
                    # Use output as input and set default output
                    input_csv = metadata.get("output_file", "")
                    self.input_text.setText(input_csv)
                    if input_csv:
                        in_path = Path(input_csv)
                        default_out = in_path.with_name(f"{in_path.stem}.spellchecked{in_path.suffix}")
                        self.output_text.setText(str(default_out))
                    self.source_metadata = metadata
                    self.log(f"✓ Loaded from {tool} output: {json_path}")
                else:
                    QMessageBox.warning(
                        self,
                        "Unknown Metadata Type",
                        f"This metadata is from '{tool}'. Expected SpellChecker or pipeline tools."
                    )
                    return
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load metadata: {str(e)}")
        
        def log(self, message: str):
            """Add message to log."""
            self.log_text.append(message)
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )
        
        def start_spell_check(self):
            """Start spell checking."""
            if not self.input_text.text():
                QMessageBox.warning(self, "Missing Input", "Please select an input file")
                return
            
            if not self.output_text.text():
                QMessageBox.warning(self, "Missing Output", "Please select an output file")
                return
            
            input_path = Path(self.input_text.text()).resolve()
            output_path = Path(self.output_text.text()).resolve()
            
            if not input_path.exists():
                QMessageBox.critical(self, "Error", f"Input file not found: {input_path}")
                return
            
            # CSV validation
            is_csv = input_path.suffix.lower() == ".csv"
            if is_csv and not self.text_col_text.text():
                QMessageBox.warning(self, "Missing Column", "Please specify the CSV text column")
                return
            
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot create output directory: {str(e)}")
                return
            
            # Clear log
            self.log_text.clear()
            self.log("=" * 60)
            self.log("Starting Spell Check")
            self.log("=" * 60)
            self.log(f"Input: {input_path}")
            self.log(f"Output: {output_path}")
            self.log(f"Mode: {self.mode_combo.currentText()}")
            self.log(f"Language: {self.lang_text.text()}")
            self.log("")
            
            # Disable controls
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.input_browse.setEnabled(False)
            self.output_browse.setEnabled(False)
            
            # Parse paths
            custom_dict_path = Path(self.custom_dict_text.text()) if self.custom_dict_text.text() else None
            ignore_list_path = Path(self.ignore_list_text.text()) if self.ignore_list_text.text() else None
            
            # Start worker
            self.worker = SpellCheckWorker(
                input_path,
                output_path,
                self.mode_combo.currentText(),
                self.lang_text.text(),
                custom_dict_path,
                ignore_list_path,
                self.text_col_text.text() if is_csv else None,
                "utf-8",
                self.source_metadata,
            )
            
            # Set format options
            self.worker.csv_format = self.csv_format_combo.currentText()
            self.worker.text_format = self.text_format_combo.currentText()
            
            self.worker.progress_update.connect(self.log)
            self.worker.correction_needed.connect(self.show_correction_dialog)
            self.worker.finished.connect(self.on_finished)
            self.worker.start()
        
        def show_correction_dialog(self, word: str, suggestions: List[str]):
            """Show correction dialog for interactive mode."""
            dialog = CorrectionDialog(word, suggestions, self)
            dialog.exec()
            if self.worker:
                self.worker.set_correction_result(dialog.result or ("ignore_once",))
        
        def stop_spell_check(self):
            """Stop spell checking."""
            if self.worker:
                self.log("Stopping...")
                self.worker.request_stop()
                self.worker.wait()
                self.on_finished(False, "Spell check stopped by user")
        
        def on_finished(self, success: bool, message: str):
            """Handle spell check completion."""
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
    window = SpellCheckerGUI()
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
