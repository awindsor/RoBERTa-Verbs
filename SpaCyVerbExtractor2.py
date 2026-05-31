#!/usr/bin/env python3
"""
SpaCy verb extractor with CSV-aware labeling and configurable context windows.

Key additions over the original extractor:
- Accepts raw text files and CSV files in the same run.
- For CSV rows, extracts text from a selected text column.
- Output can label CSV-derived rows by row number, a unique ID column, or by
  repeating every non-text column from the input row.
- The output `context` column contains a centered context window around the
  extracted verb, controlled either by character count or by an odd number of
  sentences.

The legacy extractor remains unchanged in SpaCyVerbExtractor.py.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import importlib.util
import io
import json
import logging
from logging.handlers import RotatingFileHandler
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin

import spacy
torch: Any = None
try:
    import torch as _torch
    torch = _torch
except ImportError:  # pragma: no cover - optional dependency for MPS detection
    pass


SPACY_FACTORY_DEPENDENCIES = {
    "curated_transformer": (
        "spacy-curated-transformers>=0.2.0,<0.3.0",
        "spacy_curated_transformers.pipeline.transformer",
    ),
    "transformer": (
        "spacy-transformers>=1.1.8",
        "spacy_transformers",
    ),
}


ROW_LABEL_MODE_ROW_NUMBER = "row_number"
ROW_LABEL_MODE_ID_COLUMN = "id_column"
ROW_LABEL_MODE_ALL_COLUMNS = "all_columns"
CONTEXT_MODE_SENTENCES = "sentences"
CONTEXT_MODE_CHARS = "chars"
CONTEXT_MODE_ALL = "all"


@dataclass
class SourceItem:
    source_path: Path
    source_kind: str
    text: str
    progress_bytes: int
    csv_row_number: Optional[int] = None
    csv_row_id: str = ""
    repeated_fields: Optional[Dict[str, str]] = None


@dataclass
class ExtractionStats:
    documents: int = 0
    rows_seen: int = 0
    rows_with_text: int = 0
    sentences: int = 0
    verbs: int = 0


def compute_file_md5(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def get_spacy_version_info() -> Dict[str, str]:
    return {"spacy": spacy.__version__}


def save_run_metadata(
    output_path: Path,
    input_paths: List[Path],
    settings: Dict[str, Any],
    stats: Dict[str, Any],
    command: str,
    status: str = "complete",
) -> None:
    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tool": "SpaCyVerbExtractor2",
        "status": status,
        "versions": get_spacy_version_info(),
        "input_files": [str(path) for path in input_paths],
        "input_checksums": {str(path): compute_file_md5(path) for path in input_paths if path.exists()},
        "output_file": str(output_path),
        "output_checksum": compute_file_md5(output_path) if output_path.exists() else None,
        "settings": settings,
        "statistics": stats,
        "command": command,
    }
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_run_metadata(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata JSON must contain an object: {json_path}")
    return metadata


def setup_logging(level_name: str, log_file: Path, logger_name: str = "extract_verbs2") -> logging.Logger:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=200_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def try_enable_mps(logger: Optional[logging.Logger] = None) -> bool:
    """Enable spaCy GPU mode on Apple Silicon when PyTorch MPS is available."""
    if torch is None:
        if logger:
            logger.info("PyTorch not installed; skipping MPS check.")
        return False

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        if logger:
            logger.info("PyTorch MPS backend not present; using CPU.")
        return False

    if not mps_backend.is_available():
        if logger:
            if not mps_backend.is_built():
                logger.info("PyTorch was not built with MPS support; using CPU.")
            else:
                logger.info("PyTorch MPS is unavailable on this system; using CPU.")
        return False

    try:
        spacy.require_gpu()
        if logger:
            logger.info("MPS detected; spaCy GPU mode enabled with require_gpu().")
        return True
    except Exception as exc:
        if logger:
            logger.warning(f"MPS detected but spaCy GPU activation failed: {exc}. Using CPU.")
        return False


def download_spacy_model(model_name: str, logger: Optional[logging.Logger] = None) -> None:
    if logger:
        logger.info(f"spaCy model '{model_name}' not found locally. Downloading...")

    errors: List[str] = []
    if python_has_pip():
        try:
            run_spacy_model_download(model_name, logger)
            return
        except subprocess.TimeoutExpired as exc:
            raise OSError(f"Timed out downloading spaCy model '{model_name}'.") from exc
        except OSError as exc:
            errors.append(str(exc))
            if logger:
                logger.warning(f"spaCy downloader failed: {exc}")
    elif logger:
        logger.info("Current Python environment does not include pip; trying uv instead.")

    try:
        run_uv_model_install(model_name, logger)
        return
    except subprocess.TimeoutExpired as exc:
        raise OSError(f"Timed out downloading spaCy model '{model_name}' with uv.") from exc
    except OSError as exc:
        errors.append(str(exc))

    raise OSError("\n".join(errors) or f"Failed to download spaCy model '{model_name}'.")


def python_has_pip() -> bool:
    return subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    ).returncode == 0


def install_python_requirement(requirement: str, logger: Optional[logging.Logger] = None) -> None:
    if logger:
        logger.info(f"Installing Python requirement: {requirement}")

    if python_has_pip():
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", requirement],
            capture_output=True,
            text=True,
            timeout=600,
        )
    else:
        uv_path = shutil.which("uv")
        if not uv_path:
            raise OSError(
                f"Cannot install {requirement!r}: pip is unavailable and uv was not found in PATH."
            )
        result = subprocess.run(
            [uv_path, "pip", "install", "--python", sys.executable, requirement],
            capture_output=True,
            text=True,
            timeout=600,
        )

    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        message = f"Failed to install Python requirement {requirement!r}."
        if details:
            message += f"\n{details}"
        raise OSError(message)


def run_spacy_model_download(model_name: str, logger: Optional[logging.Logger] = None) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "spacy", "download", model_name],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        message = f"Failed to download spaCy model '{model_name}'."
        if details:
            message += f"\n{details}"
        raise OSError(message)

    if logger:
        logger.info(f"Downloaded spaCy model '{model_name}' with spaCy downloader.")


def get_spacy_model_download_url(model_name: str) -> str:
    from spacy import about
    from spacy.cli.download import OLD_MODEL_SHORTCUTS, get_compatibility, get_model_filename, get_version

    package_name = OLD_MODEL_SHORTCUTS.get(model_name, model_name)
    compatibility = get_compatibility()
    version = get_version(package_name, compatibility)
    filename = get_model_filename(package_name, version)
    base_url = about.__download_url__
    if not base_url.endswith("/"):
        base_url += "/"
    download_url = urljoin(base_url, filename)
    if not download_url.startswith(about.__download_url__):
        raise OSError(f"Rejected invalid spaCy model download URL for '{model_name}'.")
    return download_url


def run_uv_model_install(model_name: str, logger: Optional[logging.Logger] = None) -> None:
    download_url = get_spacy_model_download_url(model_name)
    if logger:
        logger.info(f"Installing spaCy model '{model_name}' with uv.")

    install_python_requirement(download_url, logger)

    if logger:
        logger.info(f"Installed spaCy model '{model_name}' with uv.")


def import_spacy_factory_plugins() -> None:
    for _, module_name in SPACY_FACTORY_DEPENDENCIES.values():
        if importlib.util.find_spec(module_name.split(".", maxsplit=1)[0]) is not None:
            importlib.import_module(module_name)


def missing_factory_from_error(exc: Exception) -> Optional[str]:
    message = str(exc)
    if "[E002]" not in message:
        return None
    for factory_name in SPACY_FACTORY_DEPENDENCIES:
        if f"'{factory_name}'" in message or factory_name in message:
            return factory_name
    return None


def install_spacy_factory_dependency(factory_name: str, logger: Optional[logging.Logger] = None) -> None:
    requirement, module_name = SPACY_FACTORY_DEPENDENCIES[factory_name]
    if logger:
        logger.info(f"spaCy factory '{factory_name}' is missing; installing {requirement}.")
    install_python_requirement(requirement, logger)
    importlib.import_module(module_name)


def load_spacy_model_once(model_name: str):
    import_spacy_factory_plugins()
    return spacy.load(model_name)


def load_spacy_model(model_name: str, logger: Optional[logging.Logger] = None):
    try_enable_mps(logger)
    if logger:
        logger.info(f"Loading spaCy model: {model_name}")
    try:
        nlp = load_spacy_model_once(model_name)
    except OSError as exc:
        if "Can't find model" not in str(exc) and "No such file or directory" not in str(exc):
            raise
        try:
            download_spacy_model(model_name, logger)
        except subprocess.TimeoutExpired as timeout_exc:
            raise OSError(f"Timed out downloading spaCy model '{model_name}'.") from timeout_exc
        if logger:
            logger.info(f"Loading downloaded spaCy model: {model_name}")
        nlp = load_spacy_model_once(model_name)
    except ValueError as exc:
        factory_name = missing_factory_from_error(exc)
        if not factory_name:
            raise
        install_spacy_factory_dependency(factory_name, logger)
        if logger:
            logger.info(f"Retrying spaCy model load after installing '{factory_name}' support.")
        nlp = load_spacy_model_once(model_name)

    components_to_enable = [
        name
        for name in ["transformer", "tok2vec", "tagger", "parser", "senter", "attribute_ruler", "lemmatizer"]
        if name in nlp.pipe_names
    ]
    if components_to_enable:
        nlp.select_pipes(enable=components_to_enable)
    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
        if logger:
            logger.info("Parser not found in pipeline; added sentencizer for sentence boundaries.")
    return nlp


def iter_paths(cli_paths: List[str], paths_file: Optional[str]) -> List[Path]:
    paths = [Path(path).resolve() for path in cli_paths]
    if paths_file:
        pf = Path(paths_file)
        for raw_line in pf.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(Path(line).resolve())

    seen: set[Path] = set()
    unique_paths: List[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    return unique_paths


def expand_directory_text_files(directory: Path, filter_expr: Optional[str] = None) -> List[Path]:
    return sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.is_file()
        and not is_hidden_path(path)
        and is_raw_text_path(path)
        and file_matches_filter(filter_expr, path.resolve())
    )


def is_csv_path(path: Path) -> bool:
    return path.suffix.lower() == ".csv"


def is_raw_text_path(path: Path) -> bool:
    return path.suffix.lower() in {"", ".txt"}


def is_hidden_path(path: Path) -> bool:
    return path.name.startswith(".")


def read_csv_headers(path: Path, encoding: str) -> List[str]:
    with path.open("r", encoding=encoding, errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])


def _safe_eval_bool(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval_bool(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval_bool(elt) for elt in node.elts)
    if isinstance(node, ast.List):
        return [_safe_eval_bool(elt) for elt in node.elts]
    if isinstance(node, ast.Set):
        return {_safe_eval_bool(elt) for elt in node.elts}
    if isinstance(node, ast.Dict):
        return {
            _safe_eval_bool(k): _safe_eval_bool(v)
            for k, v in zip(node.keys or [], node.values or [])
            if k is not None
        }
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return not _safe_eval_bool(node.operand)
        if isinstance(node.op, ast.USub):
            return -_safe_eval_bool(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +_safe_eval_bool(node.operand)
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_safe_eval_bool(v) for v in node.values)
        if isinstance(node.op, ast.Or):
            return any(_safe_eval_bool(v) for v in node.values)
    if isinstance(node, ast.Compare):
        left = _safe_eval_bool(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = _safe_eval_bool(comparator)
            if isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.In):
                ok = left in right
            elif isinstance(op, ast.NotIn):
                ok = left not in right
            else:
                raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.Subscript):
        value = _safe_eval_bool(node.value)
        if isinstance(node.slice, ast.Slice):
            slice_value = slice(
                _safe_eval_bool(node.slice.lower) if node.slice.lower is not None else None,
                _safe_eval_bool(node.slice.upper) if node.slice.upper is not None else None,
                _safe_eval_bool(node.slice.step) if node.slice.step is not None else None,
            )
        else:
            slice_value = _safe_eval_bool(node.slice)
        return value[slice_value]
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def _bool_from_expression(expr: str, variables: Dict[str, str]) -> bool:
    def replace_placeholder(match: re.Match) -> str:
        key = match.group(1).strip()
        return repr(variables.get(key, ""))

    replaced = re.sub(r"\{\{\s*([^}]+?)\s*\}\}", replace_placeholder, expr)
    try:
        parsed = ast.parse(replaced, mode="eval")
        return bool(_safe_eval_bool(parsed))
    except Exception as exc:
        raise ValueError(f"Invalid filter expression: {expr!r} -> {replaced!r} ({exc})")


def file_filter_vars(path: Path) -> Dict[str, str]:
    return {
        "full path": str(path),
        "directory name": path.parent.name,
        "file name": path.name,
        "suffix": path.suffix,
    }


def file_matches_filter(expr: Optional[str], path: Path) -> bool:
    if not expr:
        return True
    return _bool_from_expression(expr, file_filter_vars(path))


def row_matches_filter(expr: Optional[str], row: Dict[str, str]) -> bool:
    if not expr:
        return True
    return _bool_from_expression(expr, row)


def collect_all_non_text_columns(paths: Sequence[Path], text_column: str, encoding: str) -> List[str]:
    columns: List[str] = []
    seen: set[str] = set()
    for path in paths:
        if not is_csv_path(path):
            continue
        for column in read_csv_headers(path, encoding):
            if column == text_column or column in seen:
                continue
            seen.add(column)
            columns.append(column)
    return columns


def count_total_input_bytes(paths: Sequence[Path]) -> int:
    """Count total input bytes for progress reporting."""
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return max(total, 1)


def validate_context_args(args: argparse.Namespace) -> None:
    if args.context_mode == CONTEXT_MODE_SENTENCES:
        if args.context_sentences < 1 or args.context_sentences % 2 == 0:
            raise SystemExit("--context-sentences must be an odd integer >= 1.")
    elif args.context_mode == CONTEXT_MODE_CHARS:
        if args.context_chars < 1:
            raise SystemExit("--context-chars must be >= 1.")
    elif args.context_mode == CONTEXT_MODE_ALL:
        return
    else:
        raise SystemExit(f"Unsupported context mode: {args.context_mode}")


def validate_csv_config(paths: Sequence[Path], args: argparse.Namespace) -> None:
    csv_paths = [path for path in paths if is_csv_path(path)]
    if not csv_paths:
        return

    if not args.csv_text_column:
        raise SystemExit("CSV input detected. Use --csv-text-column to identify the text column.")

    if args.csv_row_label_mode == ROW_LABEL_MODE_ID_COLUMN and not args.csv_id_column:
        raise SystemExit("--csv-id-column is required when --csv-row-label-mode=id_column.")

    for path in csv_paths:
        headers = read_csv_headers(path, args.encoding)
        if not headers:
            raise SystemExit(f"CSV file has no header: {path}")
        if args.csv_text_column not in headers:
            raise SystemExit(
                f"CSV file {path} is missing text column {args.csv_text_column!r}. Available: {headers}"
            )
        if args.csv_row_label_mode == ROW_LABEL_MODE_ID_COLUMN and args.csv_id_column not in headers:
            raise SystemExit(
                f"CSV file {path} is missing ID column {args.csv_id_column!r}. Available: {headers}"
            )

        if args.csv_row_label_mode == ROW_LABEL_MODE_ID_COLUMN:
            seen_ids: set[str] = set()
            with path.open("r", encoding=args.encoding, errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                for row_number, row in enumerate(reader, start=1):
                    value = (row.get(args.csv_id_column) or "").strip()
                    if not value:
                        raise SystemExit(
                            f"CSV file {path} has a blank ID in column {args.csv_id_column!r} at row {row_number}."
                        )
                    if value in seen_ids:
                        raise SystemExit(
                            f"CSV file {path} has a duplicate ID {value!r} in column {args.csv_id_column!r}."
                        )
                    seen_ids.add(value)


def normalize_input_selection(paths: Sequence[Path], filter_expr: Optional[str] = None) -> List[Path]:
    if not paths:
        raise SystemExit("No input paths provided.")

    if len(paths) != 1:
        raise SystemExit(
            "Provide exactly one input source: a directory of raw text files, a single raw text file, or a single CSV file."
        )

    selected = paths[0]
    if not selected.exists():
        raise SystemExit(f"Input path not found: {selected}")

    if selected.is_dir():
        text_files = expand_directory_text_files(selected, filter_expr)
        if not text_files:
            raise SystemExit(f"Directory contains no raw text files: {selected}")
        return text_files

    if not is_csv_path(selected) and not is_raw_text_path(selected):
        raise SystemExit(f"Input file must be .txt, .csv, or have no suffix: {selected}")

    if not is_csv_path(selected) and not file_matches_filter(filter_expr, selected):
        raise SystemExit(f"Input file does not match filter expression: {selected}")
    return [selected]


def validate_input_mode(selected_input: Path, normalized_paths: Sequence[Path]) -> None:
    if selected_input.is_dir():
        csv_files = [path for path in normalized_paths if is_csv_path(path)]
        if csv_files:
            raise SystemExit("Directory mode only supports raw text files.")
        return

    if is_csv_path(selected_input):
        if len(normalized_paths) != 1:
            raise SystemExit("CSV mode accepts exactly one CSV file.")
        return

    if len(normalized_paths) != 1:
        raise SystemExit("Raw text file mode accepts exactly one text file.")


def build_output_header(args: argparse.Namespace, repeated_columns: Sequence[str]) -> List[str]:
    if args.csv_text_column:
        header = [
            "source_kind",
            "csv_row_number",
            "csv_row_id",
            "chunk_start_char",
            "sent_start_char_in_doc",
            "sent_index_in_doc_approx",
            "token_index_in_sent",
            "lemma",
            "surface_lower",
            "span_in_context",
            "context",
        ]
    else:
        header = [
            "doc_path",
            "source_kind",
            "csv_row_number",
            "csv_row_id",
            "chunk_start_char",
            "sent_start_char_in_doc",
            "sent_index_in_doc_approx",
            "token_index_in_sent",
            "lemma",
            "surface_lower",
            "span_in_context",
            "context",
        ]
    if args.csv_row_label_mode == ROW_LABEL_MODE_ALL_COLUMNS:
        header.extend(repeated_columns)
    return header


def normalize_repeated_fields(row: Dict[str, str], repeated_columns: Sequence[str]) -> Dict[str, str]:
    return {column: row.get(column, "") for column in repeated_columns}


def iter_source_items(
    paths: Sequence[Path],
    args: argparse.Namespace,
    repeated_columns: Sequence[str],
    logger: logging.Logger,
) -> Iterable[SourceItem]:
    bytes_before = 0
    for path in paths:
        if not path.exists():
            logger.warning(f"Skipping missing path: {path}")
            continue

        try:
            path_size = path.stat().st_size
        except OSError:
            path_size = 0

        if is_csv_path(path):
            with path.open("rb") as raw_f:
                text_f = io.TextIOWrapper(raw_f, encoding=args.encoding, errors="replace", newline="")
                reader = csv.DictReader(text_f)
                for row_number, row in enumerate(reader, start=1):
                    if not row_matches_filter(args.filter_expr, row):
                        continue
                    text = row.get(args.csv_text_column, "") if args.csv_text_column else ""
                    if not text or not text.strip():
                        continue

                    repeated_fields: Optional[Dict[str, str]] = None
                    row_id = ""
                    if args.csv_row_label_mode == ROW_LABEL_MODE_ID_COLUMN and args.csv_id_column:
                        row_id = row.get(args.csv_id_column, "") or ""
                    elif args.csv_row_label_mode == ROW_LABEL_MODE_ALL_COLUMNS:
                        repeated_fields = normalize_repeated_fields(row, repeated_columns)

                    yield SourceItem(
                        source_path=path,
                        source_kind="csv",
                        text=text,
                        progress_bytes=min(bytes_before + raw_f.tell(), bytes_before + path_size),
                        csv_row_number=row_number,
                        csv_row_id=row_id,
                        repeated_fields=repeated_fields,
                    )
        else:
            text = path.read_text(encoding=args.encoding, errors="replace")
            if not text.strip():
                continue
            yield SourceItem(
                source_path=path,
                source_kind="text",
                text=text,
                progress_bytes=bytes_before + path_size,
            )
        bytes_before += path_size


def build_sentence_context(sentences: Sequence[Any], sent_index: int, text: str, size: int) -> Tuple[int, int, str]:
    half_window = size // 2
    start_index = max(0, sent_index - half_window)
    end_index = min(len(sentences), sent_index + half_window + 1)
    start_char = sentences[start_index].start_char
    end_char = sentences[end_index - 1].end_char
    return start_char, end_char, text[start_char:end_char]


def build_character_context(token: Any, text: str, window_chars: int) -> Tuple[int, int, str]:
    token_start = token.idx
    token_end = token.idx + len(token.text)
    minimum_window = max(window_chars, token_end - token_start)
    midpoint = (token_start + token_end) / 2.0
    start_char = max(0, int(round(midpoint - minimum_window / 2.0)))
    end_char = min(len(text), start_char + minimum_window)
    if end_char - start_char < minimum_window:
        start_char = max(0, end_char - minimum_window)
    return start_char, end_char, text[start_char:end_char]


def build_full_context(text: str) -> Tuple[int, int, str]:
    return 0, len(text), text


def build_context_for_token(
    token: Any,
    sent_index: int,
    sentences: Sequence[Any],
    text: str,
    args: argparse.Namespace,
) -> Tuple[str, str, int]:
    if args.context_mode == CONTEXT_MODE_SENTENCES:
        context_start, context_end, context_text = build_sentence_context(
            sentences,
            sent_index,
            text,
            args.context_sentences,
        )
    elif args.context_mode == CONTEXT_MODE_CHARS:
        context_start, context_end, context_text = build_character_context(
            token,
            text,
            args.context_chars,
        )
    elif args.context_mode == CONTEXT_MODE_ALL:
        context_start, context_end, context_text = build_full_context(text)
    else:
        raise ValueError(f"Unsupported context mode: {args.context_mode}")

    start_in_context = token.idx - context_start
    end_in_context = start_in_context + len(token.text)
    span = f"{start_in_context}-{end_in_context}"
    return span, context_text, context_start


def extract_item_rows(
    item: SourceItem,
    doc: Any,
    args: argparse.Namespace,
    repeated_columns: Sequence[str],
) -> Tuple[List[Dict[str, str]], int, int]:
    rows: List[Dict[str, str]] = []
    sentences = list(doc.sents)
    verb_count = 0

    for sent_index, sent in enumerate(sentences):
        for tok_i, tok in enumerate(sent):
            is_verb = tok.pos_ == "VERB" or (args.include_aux and tok.pos_ == "AUX")
            if not is_verb:
                continue

            verb_count += 1
            span, context_text, context_start = build_context_for_token(
                tok,
                sent_index,
                sentences,
                item.text,
                args,
            )

            row = {
                "source_kind": item.source_kind,
                "csv_row_number": str(item.csv_row_number or ""),
                "csv_row_id": item.csv_row_id,
                "chunk_start_char": str(context_start if item.source_kind == "text" else 0),
                "sent_start_char_in_doc": str(sent.start_char if item.source_kind == "text" else 0),
                "sent_index_in_doc_approx": str(sent_index),
                "token_index_in_sent": str(tok_i),
                "lemma": tok.lemma_,
                "surface_lower": tok.text.lower(),
                "span_in_context": span,
                "context": context_text,
            }
            if item.source_kind == "text":
                row["doc_path"] = str(item.source_path)

            if args.csv_row_label_mode == ROW_LABEL_MODE_ALL_COLUMNS:
                for column in repeated_columns:
                    row[column] = (item.repeated_fields or {}).get(column, "")

            rows.append(row)

    return rows, len(sentences), verb_count


def extract_to_file(
    paths: Sequence[Path],
    output_path: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    repeated_columns = []
    if args.csv_row_label_mode == ROW_LABEL_MODE_ALL_COLUMNS and args.csv_text_column:
        repeated_columns = collect_all_non_text_columns(paths, args.csv_text_column, args.encoding)

    header = build_output_header(args, repeated_columns)
    nlp = load_spacy_model(args.model, logger=logger)
    logger.info(f"spaCy pipeline: {nlp.pipe_names}")

    stats = ExtractionStats()
    start_time = time.time()
    total_bytes = count_total_input_bytes(paths)
    if progress_callback:
        progress_callback(0, total_bytes, "Preparing extraction...")

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t" if args.tsv else ",")
        writer.writeheader()

        items = iter_source_items(paths, args, repeated_columns, logger)
        for item_index, item in enumerate(items, start=1):
            if stop_check and stop_check():
                break

            stats.documents += 1 if item.source_kind == "text" else 0
            stats.rows_seen += 1 if item.source_kind == "csv" else 0
            stats.rows_with_text += 1 if item.source_kind == "csv" else 0

            label = (
                f"{item.source_path} row {item.csv_row_number}"
                if item.source_kind == "csv"
                else str(item.source_path)
            )
            emit_item_detail = (
                item.source_kind == "text"
                or args.log_every <= 1
                or item_index == 1
                or (args.log_every > 0 and item_index % args.log_every == 0)
            )
            if emit_item_detail:
                message = f"Processing {label}"
                logger.info(message)
                if progress_callback:
                    progress_callback(item.progress_bytes, total_bytes, message)

            doc = nlp(item.text)
            rows, sentence_count, verb_count = extract_item_rows(item, doc, args, repeated_columns)
            stats.sentences += sentence_count
            stats.verbs += verb_count

            for row in rows:
                writer.writerow(row)

            if progress_callback:
                status_message = (
                    f"Read {item.progress_bytes:,} of {total_bytes:,} bytes | "
                    f"sentences: {stats.sentences:,} | verbs: {stats.verbs:,}"
                )
                progress_callback(item.progress_bytes, total_bytes, status_message)

            if args.log_every > 0 and item_index % args.log_every == 0:
                progress_message = (
                    f"Progress | sources processed: {item_index:,} | "
                    f"sentences: {stats.sentences:,} | verbs: {stats.verbs:,}"
                )
                logger.info(progress_message)

    elapsed = time.time() - start_time
    summary = {
        "total_documents": stats.documents,
        "total_csv_rows": stats.rows_seen,
        "rows_with_text": stats.rows_with_text,
        "total_sentences": stats.sentences,
        "total_verbs": stats.verbs,
        "output_rows": stats.verbs,
        "elapsed_seconds": round(elapsed, 2),
    }
    return summary


def build_cli_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Extract verbs from raw text files and CSV text columns with configurable context windows."
    )
    ap.add_argument("paths", nargs="*", help="One input source: a directory, one text file, or one CSV file.")
    ap.add_argument("--paths-file", help="Text file containing one input path.")
    ap.add_argument("-o", "--output", default="verbs2.csv", help="Output CSV/TSV file.")
    ap.add_argument("--tsv", action="store_true", help="Write TSV instead of CSV.")
    ap.add_argument("--model", default=None, help="spaCy model name.")
    ap.add_argument("--encoding", default=None, help="Input file encoding.")
    ap.add_argument(
        "--filter-expr",
        default=None,
        help=(
            "CSV row filter or file-name filter using {{...}} placeholders. "
            "For CSV, use column names such as {{Speaker}} == 'Teacher' or {{grade}} in ['3', '4']. "
            "For directory file filters, use {{file name}}, {{directory name}}, {{suffix}}, or {{full path}}."
        ),
    )
    ap.add_argument("--include-aux", action="store_true", help="Treat AUX tokens as verbs.")
    ap.add_argument("--csv-text-column", default=None, help="CSV column containing the source text.")
    ap.add_argument(
        "--csv-row-label-mode",
        default=None,
        choices=[ROW_LABEL_MODE_ROW_NUMBER, ROW_LABEL_MODE_ID_COLUMN, ROW_LABEL_MODE_ALL_COLUMNS],
        help="How to label CSV-derived output rows.",
    )
    ap.add_argument("--csv-id-column", default=None, help="Unique ID column for CSV row labeling.")
    ap.add_argument(
        "--context-mode",
        default=CONTEXT_MODE_SENTENCES,
        choices=[CONTEXT_MODE_SENTENCES, CONTEXT_MODE_CHARS, CONTEXT_MODE_ALL],
        help=(
            "Whether the output context window is sentence-based, character-based, "
            "or the full source text/CSV field."
        ),
    )
    ap.add_argument("--context-sentences", type=int, default=None, help="Odd number of sentences in the context window.")
    ap.add_argument("--context-chars", type=int, default=None, help="Character width of the context window.")
    ap.add_argument("--log-every", type=int, default=None, help="Log progress every N processed sources.")
    ap.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    ap.add_argument("--load-metadata", help="Load settings and input/output paths from a SpaCyVerbExtractor2 JSON file.")
    ap.add_argument(
        "--allow-checksum-mismatch",
        action="store_true",
        help="When loading metadata, continue even if input checksums differ from the JSON.",
    )
    return ap


def reconstruct_command(args: argparse.Namespace, paths: Sequence[Path]) -> str:
    cmd = ["python", "SpaCyVerbExtractor2.py"]
    cmd.extend(str(path) for path in paths)
    if getattr(args, "load_metadata", None):
        cmd.extend(["--load-metadata", args.load_metadata])
    if getattr(args, "allow_checksum_mismatch", False):
        cmd.append("--allow-checksum-mismatch")
    if args.paths_file:
        cmd.extend(["--paths-file", args.paths_file])
    if args.output != "verbs2.csv":
        cmd.extend(["--output", args.output])
    if args.tsv:
        cmd.append("--tsv")
    if args.model != "en_core_web_sm":
        cmd.extend(["--model", args.model])
    if args.encoding != "utf-8":
        cmd.extend(["--encoding", args.encoding])
    if args.filter_expr:
        cmd.extend(["--filter-expr", args.filter_expr])
    if args.include_aux:
        cmd.append("--include-aux")
    if args.csv_text_column:
        cmd.extend(["--csv-text-column", args.csv_text_column])
    if args.csv_row_label_mode != ROW_LABEL_MODE_ROW_NUMBER:
        cmd.extend(["--csv-row-label-mode", args.csv_row_label_mode])
    if args.csv_id_column:
        cmd.extend(["--csv-id-column", args.csv_id_column])
    if args.context_mode != CONTEXT_MODE_SENTENCES:
        cmd.extend(["--context-mode", args.context_mode])
    if args.context_sentences != 1:
        cmd.extend(["--context-sentences", str(args.context_sentences)])
    if args.context_chars != 301:
        cmd.extend(["--context-chars", str(args.context_chars)])
    if args.log_every != 10000:
        cmd.extend(["--log-every", str(args.log_every)])
    if args.log_level != "INFO":
        cmd.extend(["--log-level", args.log_level])
    return " ".join(cmd)


def metadata_settings_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "model": args.model,
        "encoding": args.encoding,
        "filter_expr": args.filter_expr,
        "include_aux": args.include_aux,
        "csv_text_column": args.csv_text_column,
        "csv_row_label_mode": args.csv_row_label_mode,
        "csv_id_column": args.csv_id_column,
        "context_mode": args.context_mode,
        "context_sentences": args.context_sentences,
        "context_chars": args.context_chars,
        "output_format": "tsv" if args.tsv else "csv",
    }


def metadata_cli_paths(metadata: Dict[str, Any]) -> List[str]:
    input_paths = [Path(path) for path in metadata.get("input_files", [])]
    if len(input_paths) <= 1:
        return [str(path) for path in input_paths]

    parents = {path.parent for path in input_paths}
    if len(parents) == 1 and all(is_raw_text_path(path) for path in input_paths):
        return [str(next(iter(parents)))]

    return [str(path) for path in input_paths]


def verify_metadata_input_checksums(paths: Sequence[Path], metadata: Dict[str, Any]) -> Dict[str, List[str]]:
    input_checksums = metadata.get("input_checksums", {})
    checksum_lookup = dict(input_checksums)
    for saved_path, checksum in input_checksums.items():
        try:
            checksum_lookup[str(Path(saved_path).resolve())] = checksum
        except OSError:
            pass
    issues: Dict[str, List[str]] = {
        "missing_files": [],
        "checksum_mismatches": [],
        "unverified_files": [],
    }

    for path in paths:
        path_str = str(path)
        expected_checksum = checksum_lookup.get(path_str)
        if expected_checksum is None:
            expected_checksum = checksum_lookup.get(str(path.resolve()))
        if expected_checksum is None:
            issues["unverified_files"].append(path_str)
            continue
        if not path.exists():
            issues["missing_files"].append(path_str)
            continue
        actual_checksum = compute_file_md5(path)
        if actual_checksum != expected_checksum:
            issues["checksum_mismatches"].append(path_str)

    return issues


def format_checksum_issues(issues: Dict[str, List[str]]) -> str:
    messages = []
    if issues["missing_files"]:
        messages.append("Missing input files:\n" + "\n".join(issues["missing_files"]))
    if issues["checksum_mismatches"]:
        messages.append("Input files changed since metadata was written:\n" + "\n".join(issues["checksum_mismatches"]))
    if issues["unverified_files"]:
        messages.append("Input files not listed in metadata:\n" + "\n".join(issues["unverified_files"]))
    return "\n\n".join(messages)


def load_cli_metadata_defaults(args: argparse.Namespace, provided_args: set[str]) -> Optional[Dict[str, Any]]:
    if not args.load_metadata:
        return None

    metadata_path = Path(args.load_metadata)
    if not metadata_path.exists():
        raise SystemExit(f"Metadata file not found: {metadata_path}")

    metadata = load_run_metadata(metadata_path)
    tool_name = metadata.get("tool", "unknown")
    if tool_name != "SpaCyVerbExtractor2":
        raise SystemExit(
            f"Unsupported metadata source: {tool_name!r}. Expected 'SpaCyVerbExtractor2'."
        )

    settings = metadata.get("settings", {})
    if not args.paths and not args.paths_file:
        args.paths = metadata_cli_paths(metadata)
    if args.output == "verbs2.csv" and "-o" not in provided_args and "--output" not in provided_args:
        args.output = metadata.get("output_file") or args.output
    if args.model is None and "--model" not in provided_args:
        args.model = settings.get("model")
    if args.encoding is None and "--encoding" not in provided_args:
        args.encoding = settings.get("encoding")
    if args.filter_expr is None and "--filter-expr" not in provided_args:
        args.filter_expr = settings.get("filter_expr")
    if not args.include_aux and "--include-aux" not in provided_args:
        args.include_aux = bool(settings.get("include_aux", False))
    if args.csv_text_column is None and "--csv-text-column" not in provided_args:
        args.csv_text_column = settings.get("csv_text_column")
    if args.csv_row_label_mode is None and "--csv-row-label-mode" not in provided_args:
        args.csv_row_label_mode = settings.get("csv_row_label_mode")
    if args.csv_id_column is None and "--csv-id-column" not in provided_args:
        args.csv_id_column = settings.get("csv_id_column")
    if args.context_mode == CONTEXT_MODE_SENTENCES and "--context-mode" not in provided_args:
        args.context_mode = settings.get("context_mode", args.context_mode)
    if args.context_sentences is None and "--context-sentences" not in provided_args:
        args.context_sentences = settings.get("context_sentences")
    if args.context_chars is None and "--context-chars" not in provided_args:
        args.context_chars = settings.get("context_chars")
    if not args.tsv and "--tsv" not in provided_args:
        args.tsv = settings.get("output_format") == "tsv"
    return metadata


def apply_cli_defaults(args: argparse.Namespace) -> None:
    args.model = args.model or "en_core_web_sm"
    args.encoding = args.encoding or "utf-8"
    args.csv_row_label_mode = args.csv_row_label_mode or ROW_LABEL_MODE_ROW_NUMBER
    args.context_sentences = int(args.context_sentences or 1)
    args.context_chars = int(args.context_chars or 301)
    args.log_every = int(args.log_every or 10000)
    args.log_level = args.log_level or "INFO"


def run_cli() -> None:
    parser = build_cli_parser()
    args = parser.parse_args()
    metadata = load_cli_metadata_defaults(args, set(sys.argv[1:]))
    apply_cli_defaults(args)
    validate_context_args(args)

    selected_paths = iter_paths(args.paths, args.paths_file)
    normalized_paths = normalize_input_selection(selected_paths, args.filter_expr)
    validate_input_mode(selected_paths[0], normalized_paths)
    if metadata:
        checksum_issues = verify_metadata_input_checksums(normalized_paths, metadata)
        has_checksum_issues = any(checksum_issues.values())
        if has_checksum_issues and not args.allow_checksum_mismatch:
            raise SystemExit(
                format_checksum_issues(checksum_issues)
                + "\n\nUse --allow-checksum-mismatch to run anyway."
            )
        if has_checksum_issues:
            print("Warning: " + format_checksum_issues(checksum_issues))

    validate_csv_config(normalized_paths, args)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.log_level, output_path.with_suffix(".log"))

    command = reconstruct_command(args, selected_paths)
    save_run_metadata(
        output_path,
        list(normalized_paths),
        settings=metadata_settings_from_args(args),
        stats={"message": "Run configured; extraction has not started."},
        command=command,
        status="started",
    )
    logger.info(f"Wrote preliminary metadata: {output_path.with_suffix('.json')}")

    try:
        stats = extract_to_file(normalized_paths, output_path, args, logger)
    except Exception:
        save_run_metadata(
            output_path,
            list(normalized_paths),
            settings=metadata_settings_from_args(args),
            stats={"message": "Run failed before completion."},
            command=command,
            status="failed",
        )
        raise

    save_run_metadata(
        output_path,
        list(normalized_paths),
        settings=metadata_settings_from_args(args),
        stats=stats,
        command=command,
        status="complete",
    )
    logger.info(f"Wrote output: {output_path}")
    logger.info(f"Wrote metadata: {output_path.with_suffix('.json')}")


def run_gui() -> None:
    try:
        from PySide6.QtCore import QThread, Signal
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QProgressBar,
            QRadioButton,
            QScrollArea,
            QSpinBox,
            QTextEdit,
            QVBoxLayout,
            QWidget,
            QSizePolicy,
        )
    except ImportError:
        print("Error: PySide6 is required for GUI mode.")
        print("Install it with: pip install PySide6")
        sys.exit(1)

    class ExtractionWorker(QThread):
        progress_update = Signal(int, int, str)
        finished = Signal(bool, str)

        def __init__(self, paths: List[Path], output_path: Path, args: argparse.Namespace):
            super().__init__()
            self.paths = paths
            self.output_path = output_path
            self.args = args
            self._stop_requested = False

        def request_stop(self) -> None:
            self._stop_requested = True

        def run(self) -> None:
            logger = setup_logging(
                self.args.log_level,
                self.output_path.with_suffix(".log"),
                logger_name="extract_verbs2_gui",
            )
            try:
                command = reconstruct_command(self.args, self.paths)
                save_run_metadata(
                    self.output_path,
                    self.paths,
                    settings=metadata_settings_from_args(self.args),
                    stats={"message": "Run configured; extraction has not started."},
                    command=command,
                    status="started",
                )
                logger.info(f"Wrote preliminary metadata: {self.output_path.with_suffix('.json')}")

                stats = extract_to_file(
                    self.paths,
                    self.output_path,
                    self.args,
                    logger,
                    progress_callback=self.progress_update.emit,
                    stop_check=lambda: self._stop_requested,
                )
                save_run_metadata(
                    self.output_path,
                    self.paths,
                    settings=metadata_settings_from_args(self.args),
                    stats=stats,
                    command=command,
                    status="complete",
                )
                message = f"Extraction complete. Output: {self.output_path}"
                self.progress_update.emit(1, 1, message)
                self.finished.emit(True, message)
            except Exception as exc:
                save_run_metadata(
                    self.output_path,
                    self.paths,
                    settings=metadata_settings_from_args(self.args),
                    stats={"message": f"Run failed before completion: {exc}"},
                    command=reconstruct_command(self.args, self.paths),
                    status="failed",
                )
                self.finished.emit(False, f"Error: {exc}")

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("SpaCy Verb Extractor 2")
            self.resize(1100, 760)
            self.input_paths: List[Path] = []
            self.worker: Optional[ExtractionWorker] = None
            self.csv_columns: List[str] = []
            self.init_ui()

        def init_ui(self) -> None:
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            self.setCentralWidget(scroll_area)
            central = QWidget()
            scroll_area.setWidget(central)
            outer = QVBoxLayout(central)

            files_group = QGroupBox("Input Files")
            files_layout = QVBoxLayout(files_group)
            file_buttons = QHBoxLayout()
            self.add_files_btn = QPushButton("Add Files")
            self.add_files_btn.clicked.connect(self.add_files)
            self.add_folder_btn = QPushButton("Add Folder")
            self.add_folder_btn.clicked.connect(self.add_folder)
            self.add_paths_btn = QPushButton("Add Paths File")
            self.add_paths_btn.clicked.connect(self.add_paths_file)
            self.load_json_btn = QPushButton("Load JSON")
            self.load_json_btn.clicked.connect(self.load_json)
            self.clear_btn = QPushButton("Clear")
            self.clear_btn.clicked.connect(self.clear_files)
            file_buttons.addWidget(self.add_files_btn)
            file_buttons.addWidget(self.add_folder_btn)
            file_buttons.addWidget(self.add_paths_btn)
            file_buttons.addWidget(self.load_json_btn)
            file_buttons.addWidget(self.clear_btn)
            file_buttons.addStretch()
            files_layout.addLayout(file_buttons)
            self.files_list = QListWidget()
            self.files_list.setMaximumHeight(120)
            files_layout.addWidget(self.files_list)
            input_options_row = QHBoxLayout()
            input_options_row.addWidget(QLabel("Encoding:"))
            self.encoding_combo = QComboBox()
            self.encoding_combo.addItems(["utf-8", "utf-8-sig", "latin-1", "cp1252"])
            input_options_row.addWidget(self.encoding_combo)
            input_options_row.addStretch()
            files_layout.addLayout(input_options_row)
            filter_row = QHBoxLayout()
            filter_row.addWidget(QLabel("Filter:"))
            self.filter_edit = QLineEdit()
            self.filter_edit.setPlaceholderText("Example: {{Speaker}} == 'Teacher'")
            self.filter_edit.setToolTip(
                "CSV filters use column placeholders, for example {{Speaker}} == 'Teacher' "
                "or {{grade}} in ['3', '4']. Directory filters can use {{file name}}, "
                "{{directory name}}, {{suffix}}, or {{full path}}."
            )
            filter_row.addWidget(self.filter_edit)
            files_layout.addLayout(filter_row)
            filter_help = QLabel(
                "CSV filter syntax: {{column name}} == 'value'; supports and/or, !=, <, >, in, not in. "
                "Directory filters can use {{file name}}, {{directory name}}, {{suffix}}, or {{full path}}."
            )
            filter_help.setWordWrap(True)
            files_layout.addWidget(filter_help)
            outer.addWidget(files_group)

            settings_group = QGroupBox("Settings")
            settings_layout = QFormLayout(settings_group)

            spacy_group = QGroupBox("SpaCy")
            spacy_layout = QHBoxLayout(spacy_group)
            spacy_layout.addWidget(QLabel("SpaCy Model:"))
            self.model_combo = QComboBox()
            self.model_combo.addItems(["en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"])
            spacy_layout.addWidget(self.model_combo, 1)
            self.include_aux_check = QCheckBox("Process AUX tokens")
            spacy_layout.addWidget(self.include_aux_check)
            settings_layout.addRow(spacy_group)

            output_group = QGroupBox("Output")
            output_layout = QGridLayout(output_group)
            self.output_input = QLineEdit("verbs2.csv")
            self.output_input.setMinimumWidth(300)
            self.output_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.output_browse_btn = QPushButton("Browse...")
            self.output_browse_btn.clicked.connect(self.browse_output)
            self.output_browse_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            output_layout.addWidget(QLabel("Output File:"), 0, 0)
            output_file_row = QHBoxLayout()
            output_file_row.addWidget(self.output_input, 1)
            output_file_row.addWidget(self.output_browse_btn, 0)
            output_layout.addWidget(self._wrap_layout(output_file_row), 0, 1, 1, 3)

            self.text_column_combo = QComboBox()
            self.text_column_combo.setEditable(True)
            output_layout.addWidget(QLabel("CSV Text Column:"), 1, 0)
            output_layout.addWidget(self.text_column_combo, 1, 1)

            self.row_label_combo = QComboBox()
            self.row_label_combo.currentIndexChanged.connect(self.on_row_label_mode_changed)
            output_layout.addWidget(QLabel("CSV Row Label:"), 1, 2)
            output_layout.addWidget(self.row_label_combo, 1, 3)

            self.id_column_combo = QComboBox()
            self.id_column_combo.setEditable(True)
            output_layout.addWidget(QLabel("CSV ID Column:"), 2, 0)
            output_layout.addWidget(self.id_column_combo, 2, 1)

            self.tsv_check = QCheckBox("Write TSV")
            output_layout.addWidget(self.tsv_check, 2, 2, 1, 2)
            output_layout.setColumnStretch(1, 1)
            output_layout.setColumnStretch(3, 1)
            settings_layout.addRow(output_group)

            context_group = QGroupBox("Context")
            context_layout = QGridLayout(context_group)
            self.context_sent_radio = QRadioButton()
            self.context_sent_radio.toggled.connect(self.on_context_mode_changed)
            self.context_char_radio = QRadioButton()
            self.context_char_radio.toggled.connect(self.on_context_mode_changed)
            self.context_all_radio = QRadioButton()
            self.context_all_radio.toggled.connect(self.on_context_mode_changed)
            self.context_sent_spin = QSpinBox()
            self.context_sent_spin.setMinimum(1)
            self.context_sent_spin.setMaximum(99)
            self.context_sent_spin.setSingleStep(2)
            self.context_sent_spin.setValue(1)
            self.context_char_spin = QSpinBox()
            self.context_char_spin.setMinimum(1)
            self.context_char_spin.setMaximum(20000)
            self.context_char_spin.setValue(301)
            context_layout.addWidget(self.context_sent_radio, 0, 0)
            context_layout.addWidget(QLabel("Sentences"), 0, 1)
            context_layout.addWidget(QLabel("Context Size:"), 0, 2)
            context_layout.addWidget(self.context_sent_spin, 0, 3)
            context_layout.addWidget(self.context_char_radio, 1, 0)
            context_layout.addWidget(QLabel("Characters"), 1, 1)
            context_layout.addWidget(QLabel("Context Size:"), 1, 2)
            context_layout.addWidget(self.context_char_spin, 1, 3)
            context_layout.addWidget(self.context_all_radio, 2, 0)
            context_layout.addWidget(QLabel("All"), 2, 1)
            context_layout.addWidget(QLabel("Full source text/CSV field"), 2, 2, 1, 2)
            context_layout.setColumnStretch(1, 1)
            context_layout.setColumnStretch(3, 1)
            self.context_sent_radio.setChecked(True)
            settings_layout.addRow(context_group)
            outer.addWidget(settings_group)

            progress_group = QGroupBox("Progress")
            progress_layout = QVBoxLayout(progress_group)
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 10000)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("%p%")
            self.progress_bar.setVisible(False)
            progress_layout.addWidget(self.progress_bar)
            self.progress_status = QLabel("Ready")
            progress_layout.addWidget(self.progress_status)
            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            progress_layout.addWidget(self.log_text)
            outer.addWidget(progress_group)

            buttons = QHBoxLayout()
            buttons.addStretch()
            self.start_btn = QPushButton("Start Extraction")
            self.start_btn.clicked.connect(self.start_extraction)
            self.stop_btn = QPushButton("Stop")
            self.stop_btn.clicked.connect(self.stop_extraction)
            self.stop_btn.setEnabled(False)
            buttons.addWidget(self.start_btn)
            buttons.addWidget(self.stop_btn)
            buttons.addStretch()
            outer.addLayout(buttons)

            self.refresh_csv_controls()
            self.on_context_mode_changed()

        def _wrap_layout(self, layout):
            wrapper = QWidget()
            wrapper.setLayout(layout)
            return wrapper

        def log(self, message: str) -> None:
            self.log_text.append(message)
            self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

        def update_progress(self, current: int, total: int, status: str) -> None:
            progress_max = self.progress_bar.maximum()
            progress = progress_max if total <= 0 else min(progress_max, int((current / total) * progress_max))
            percent = 100.0 if total <= 0 else min(100.0, (current / total) * 100)
            self.progress_bar.setValue(progress)
            self.progress_bar.setFormat(f"{percent:.1f}%")
            self.progress_status.setText(status)

        def add_files(self) -> None:
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Input Files",
                "",
                "Text, CSV, and Suffixless Files (*.txt *.csv *);;All Files (*)",
            )
            for raw_path in files:
                self.set_single_input(Path(raw_path).resolve())
            self.refresh_csv_controls()

        def add_folder(self) -> None:
            directory = QFileDialog.getExistingDirectory(
                self,
                "Select Folder of Raw Text Files",
                "",
            )
            if directory:
                self.set_single_input(Path(directory).resolve())
                self.refresh_csv_controls()

        def add_paths_file(self) -> None:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Select Paths File",
                "",
                "Text Files (*.txt);;All Files (*)",
            )
            if not file_name:
                return
            selected = iter_paths([], file_name)
            if not selected:
                return
            self.set_single_input(selected[0])
            self.refresh_csv_controls()

        def load_json(self) -> None:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Select Metadata JSON",
                "",
                "JSON Files (*.json);;All Files (*)",
            )
            if file_name:
                self.load_json_from_file(Path(file_name))

        def load_json_from_file(self, json_path: Path) -> None:
            try:
                metadata = load_run_metadata(json_path)
                tool_name = metadata.get("tool", "unknown")
                if tool_name != "SpaCyVerbExtractor2":
                    QMessageBox.warning(
                        self,
                        "Unsupported JSON",
                        f"This JSON is from {tool_name!r}; expected 'SpaCyVerbExtractor2'.",
                    )
                    return

                input_paths = [Path(path) for path in metadata.get("input_files", [])]
                warnings: List[str] = []
                for path_str, expected_checksum in metadata.get("input_checksums", {}).items():
                    path = Path(path_str)
                    if not path.exists():
                        warnings.append(f"Missing input file: {path}")
                    elif compute_file_md5(path) != expected_checksum:
                        warnings.append(f"Input file changed: {path}")

                if warnings:
                    message = "\n".join(warnings[:10])
                    if len(warnings) > 10:
                        message += f"\n...and {len(warnings) - 10} more."
                    message += "\n\nLoad settings anyway?"
                    reply = QMessageBox.warning(
                        self,
                        "Input File Issues",
                        message,
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )
                    if reply == QMessageBox.StandardButton.No:
                        return

                settings = metadata.get("settings", {})
                selected_input = self.metadata_input_source(input_paths)
                self.input_paths.clear()
                self.files_list.clear()
                if selected_input:
                    self.set_single_input(selected_input)

                self.output_input.setText(metadata.get("output_file", "verbs2.csv"))
                self.model_combo.setCurrentText(settings.get("model", "en_core_web_sm"))
                self.encoding_combo.setCurrentText(settings.get("encoding", "utf-8"))
                self.filter_edit.setText(settings.get("filter_expr") or "")
                self.include_aux_check.setChecked(bool(settings.get("include_aux", False)))
                self.tsv_check.setChecked(settings.get("output_format", "csv") == "tsv")

                self.refresh_csv_controls()
                self.text_column_combo.setEditText(settings.get("csv_text_column") or "")
                self.set_combo_data(
                    self.row_label_combo,
                    settings.get("csv_row_label_mode", ROW_LABEL_MODE_ROW_NUMBER),
                )
                self.id_column_combo.setEditText(settings.get("csv_id_column") or "")
                self.on_row_label_mode_changed()

                context_mode = settings.get("context_mode", CONTEXT_MODE_SENTENCES)
                self.context_all_radio.setChecked(context_mode == CONTEXT_MODE_ALL)
                self.context_char_radio.setChecked(context_mode == CONTEXT_MODE_CHARS)
                self.context_sent_radio.setChecked(context_mode == CONTEXT_MODE_SENTENCES)
                self.context_sent_spin.setValue(int(settings.get("context_sentences", 1)))
                self.context_char_spin.setValue(int(settings.get("context_chars", 301)))
                self.on_context_mode_changed()

                self.log(f"Loaded settings from: {json_path}")
                for warning in warnings[:10]:
                    self.log(warning)
            except Exception as exc:
                QMessageBox.critical(self, "JSON Load Error", f"Failed to load JSON: {exc}")

        def metadata_input_source(self, paths: Sequence[Path]) -> Optional[Path]:
            if not paths:
                return None
            if len(paths) == 1:
                return paths[0]
            parents = {path.parent for path in paths}
            if len(parents) == 1 and all(is_raw_text_path(path) for path in paths):
                return next(iter(parents))
            self.log("JSON lists multiple input files; settings loaded, but input source was not restored.")
            return None

        def set_combo_data(self, combo: QComboBox, value: str) -> None:
            index = combo.findData(value)
            if index >= 0:
                combo.setCurrentIndex(index)

        def clear_files(self) -> None:
            self.input_paths.clear()
            self.files_list.clear()
            self.refresh_csv_controls()

        def set_single_input(self, path: Path) -> None:
            self.input_paths = [path]
            self.files_list.clear()
            label = f"{path} [directory]" if path.is_dir() else str(path)
            self.files_list.addItem(label)

        def browse_output(self) -> None:
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Select Output File",
                "verbs2.csv",
                "CSV Files (*.csv);;TSV Files (*.tsv);;All Files (*)",
            )
            if file_name:
                self.output_input.setText(file_name)

        def refresh_csv_controls(self) -> None:
            columns: List[str] = []
            seen: set[str] = set()
            encoding = self.encoding_combo.currentText() or "utf-8"
            for path in self.input_paths:
                if not is_csv_path(path) or not path.exists():
                    continue
                try:
                    for column in read_csv_headers(path, encoding):
                        if column not in seen:
                            seen.add(column)
                            columns.append(column)
                except Exception:
                    continue

            self.csv_columns = columns
            current_text = self.text_column_combo.currentText()
            current_id = self.id_column_combo.currentText()

            self.text_column_combo.blockSignals(True)
            self.id_column_combo.blockSignals(True)
            self.row_label_combo.blockSignals(True)

            self.text_column_combo.clear()
            self.text_column_combo.addItems(columns)
            if current_text:
                self.text_column_combo.setEditText(current_text)

            self.id_column_combo.clear()
            self.id_column_combo.addItems(columns)
            if current_id:
                self.id_column_combo.setEditText(current_id)

            self.row_label_combo.clear()
            self.row_label_combo.addItem("Row Number", ROW_LABEL_MODE_ROW_NUMBER)
            self.row_label_combo.addItem("Unique ID Column", ROW_LABEL_MODE_ID_COLUMN)
            self.row_label_combo.addItem("All Non-Text Columns", ROW_LABEL_MODE_ALL_COLUMNS)

            has_csv = bool(columns)
            self.text_column_combo.setEnabled(has_csv)
            self.row_label_combo.setEnabled(has_csv)
            self.id_column_combo.setEnabled(has_csv)

            self.text_column_combo.blockSignals(False)
            self.id_column_combo.blockSignals(False)
            self.row_label_combo.blockSignals(False)
            self.on_row_label_mode_changed()

        def on_row_label_mode_changed(self) -> None:
            mode = self.row_label_combo.currentData()
            self.id_column_combo.setEnabled(mode == ROW_LABEL_MODE_ID_COLUMN and self.row_label_combo.isEnabled())

        def on_context_mode_changed(self) -> None:
            if self.context_all_radio.isChecked():
                mode = CONTEXT_MODE_ALL
            elif self.context_char_radio.isChecked():
                mode = CONTEXT_MODE_CHARS
            else:
                mode = CONTEXT_MODE_SENTENCES
            self.context_sent_spin.setEnabled(mode == CONTEXT_MODE_SENTENCES)
            self.context_char_spin.setEnabled(mode == CONTEXT_MODE_CHARS)

        def build_args(self) -> argparse.Namespace:
            if self.context_all_radio.isChecked():
                mode = CONTEXT_MODE_ALL
            elif self.context_char_radio.isChecked():
                mode = CONTEXT_MODE_CHARS
            else:
                mode = CONTEXT_MODE_SENTENCES
            return argparse.Namespace(
                output=self.output_input.text(),
                tsv=self.tsv_check.isChecked(),
                model=self.model_combo.currentText(),
                encoding=self.encoding_combo.currentText(),
                filter_expr=self.filter_edit.text().strip() or None,
                include_aux=self.include_aux_check.isChecked(),
                csv_text_column=self.text_column_combo.currentText().strip() or None,
                csv_row_label_mode=self.row_label_combo.currentData() or ROW_LABEL_MODE_ROW_NUMBER,
                csv_id_column=self.id_column_combo.currentText().strip() or None,
                context_mode=mode,
                context_sentences=self.context_sent_spin.value(),
                context_chars=self.context_char_spin.value(),
                log_every=10000,
                log_level="INFO",
                paths_file=None,
                paths=[],
            )

        def start_extraction(self) -> None:
            if not self.input_paths:
                QMessageBox.warning(self, "No Input Files", "Please select one input source.")
                return

            args = self.build_args()
            try:
                validate_context_args(args)
                normalized_paths = normalize_input_selection(self.input_paths, args.filter_expr)
                validate_input_mode(self.input_paths[0], normalized_paths)
                validate_csv_config(normalized_paths, args)
            except SystemExit as exc:
                QMessageBox.critical(self, "Configuration Error", str(exc))
                return

            output_path = Path(args.output).resolve()
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                QMessageBox.critical(self, "Output Error", f"Cannot create output directory: {exc}")
                return

            self.log_text.clear()
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_status.setText("Starting...")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.worker = ExtractionWorker(list(normalized_paths), output_path, args)
            self.worker.progress_update.connect(self.update_progress)
            self.worker.finished.connect(self.on_finished)
            self.worker.start()

        def stop_extraction(self) -> None:
            if self.worker:
                self.progress_status.setText("Stopping...")
                self.worker.request_stop()

        def on_finished(self, success: bool, message: str) -> None:
            self.progress_bar.setVisible(False)
            self.progress_status.setText(message)
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            if success:
                QMessageBox.information(self, "Success", message)
            else:
                self.log(message)
                QMessageBox.critical(self, "Error", message)

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
