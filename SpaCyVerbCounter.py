#!/usr/bin/env python3
"""
Aggregate verb frequencies from extract_verbs CSV (generator-based).

Usage:
  python verb_freqs.py input.csv output.csv --field lemma
  python verb_freqs.py input.csv output.csv --field surface_form
"""

import csv
import argparse
import sys
from collections import Counter
from typing import Iterator


def stream_column(path: str, field: str) -> Iterator[str]:
    """Yield values from a single column, one row at a time."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row[field]


def count_and_write(input_csv: str, output_csv: str, field: str) -> int:
    """Count a field and write the frequency table. Returns unique-token count."""
    counter = Counter(stream_column(input_csv, field))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([field, "freq"])
        for token, freq in counter.most_common():
            writer.writerow([token, freq])

    return len(counter)


def run_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("output_csv")
    ap.add_argument(
        "--field",
        choices=["lemma", "surface_lower"],
        required=True
    )
    args = ap.parse_args()
    count_and_write(args.input_csv, args.output_csv, args.field)


def run_gui() -> None:
    try:
        from PySide6.QtWidgets import (
            QApplication,
            QComboBox,
            QFileDialog,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        print("Error: PySide6 is required for GUI mode.")
        print("Install it with: pip install PySide6")
        sys.exit(1)

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("SpaCy Verb Counter")
            self.resize(700, 320)
            self.init_ui()

        def init_ui(self) -> None:
            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)

            input_row = QHBoxLayout()
            input_row.addWidget(QLabel("Input CSV:"))
            self.input_edit = QLineEdit()
            input_row.addWidget(self.input_edit)
            input_btn = QPushButton("Browse...")
            input_btn.clicked.connect(self.browse_input)
            input_row.addWidget(input_btn)
            layout.addLayout(input_row)

            output_row = QHBoxLayout()
            output_row.addWidget(QLabel("Output CSV:"))
            self.output_edit = QLineEdit("verb_counts.csv")
            output_row.addWidget(self.output_edit)
            output_btn = QPushButton("Browse...")
            output_btn.clicked.connect(self.browse_output)
            output_row.addWidget(output_btn)
            layout.addLayout(output_row)

            field_row = QHBoxLayout()
            field_row.addWidget(QLabel("Field:"))
            self.field_combo = QComboBox()
            self.field_combo.addItems(["lemma", "surface_lower"])
            field_row.addWidget(self.field_combo)
            field_row.addStretch()
            layout.addLayout(field_row)

            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            layout.addWidget(self.log_text)

            button_row = QHBoxLayout()
            button_row.addStretch()
            run_btn = QPushButton("Run")
            run_btn.clicked.connect(self.run_counter)
            button_row.addWidget(run_btn)
            layout.addLayout(button_row)

        def browse_input(self) -> None:
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Select Input CSV", "", "CSV Files (*.csv);;All Files (*)"
            )
            if file_name:
                self.input_edit.setText(file_name)

        def browse_output(self) -> None:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Select Output CSV", "verb_counts.csv", "CSV Files (*.csv);;All Files (*)"
            )
            if file_name:
                self.output_edit.setText(file_name)

        def log(self, message: str) -> None:
            self.log_text.append(message)

        def run_counter(self) -> None:
            input_csv = self.input_edit.text().strip()
            output_csv = self.output_edit.text().strip()
            field = self.field_combo.currentText()

            if not input_csv or not output_csv:
                QMessageBox.warning(self, "Missing Input", "Please provide input and output CSV paths.")
                return

            try:
                unique_count = count_and_write(input_csv, output_csv, field)
                self.log(f"Wrote frequency table to {output_csv}")
                self.log(f"Unique {field} values: {unique_count}")
                QMessageBox.information(self, "Success", f"Finished writing {output_csv}")
            except Exception as exc:
                self.log(f"Error: {exc}")
                QMessageBox.critical(self, "Error", str(exc))

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


def main() -> None:
    if len(sys.argv) == 1:
        run_gui()
    else:
        run_cli()


if __name__ == "__main__":
    main()
