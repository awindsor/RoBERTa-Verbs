# SpaCyVerbExtractor Metadata Features - Implementation Summary

## Overview

SpaCyVerbExtractor.py now includes comprehensive metadata support for complete run reconstruction. Both CLI and GUI modes can save, load, and verify extraction runs.

## New Features

### 1. Automatic Metadata Generation

When an extraction completes, a JSON metadata file is created with the same base name as the output:
- `verbs.csv` → `verbs.json`
- `output.tsv` → `output.json`

**Metadata includes:**
```json
{
  "timestamp": "2026-01-24T15:30:45.123456Z",
  "output_file": "/path/to/verbs.csv",
  "output_checksum": "a1b2c3d4e5f6...",
  "input_files": ["/path/file1.txt", "/path/file2.txt"],
  "input_checksums": {
    "/path/file1.txt": "f1e2d3c4b5a6...",
    "/path/file2.txt": "6b5a4c3d2e1f..."
  },
  "settings": {
    "model": "en_core_web_sm",
    "encoding": "utf-8",
    "include_aux": false,
    "chunk_size": 2000000,
    "overlap": 5000,
    "dedupe_window": 50000,
    "heartbeat_chunks": 10,
    "output_format": "csv"
  },
  "statistics": {
    "total_documents": 5,
    "total_chunks": 150,
    "total_sentences": 10000,
    "total_verbs": 45000,
    "elapsed_seconds": 234.56
  }
}
```

### 2. CLI Mode: Load and Override Settings

**Load settings from previous run:**
```bash
python SpaCyVerbExtractor.py --load-metadata verbs.json
```

**Load settings but override with command-line arguments:**
```bash
# Loads all settings from verbs.json, but uses chunk-size from CLI
python SpaCyVerbExtractor.py --load-metadata verbs.json --chunk-size 1000000

# Loads from JSON, adds new input files
python SpaCyVerbExtractor.py --load-metadata verbs.json new_input.txt -o new_output.csv
```

**Exact run reconstruction:**
```bash
# Recreate exact extraction with same settings
python SpaCyVerbExtractor.py --load-metadata verbs.json
```

### 3. GUI Mode: Load and Verify

**"Load Settings from JSON" button:**
1. Opens file dialog to select metadata JSON
2. Verifies input file checksums against saved checksums
3. Displays warnings for:
   - Missing input files
   - Changed input files (checksum mismatch)
4. Asks user to confirm before proceeding
5. Pre-populates all GUI fields with saved settings
6. User can modify any setting before running

**Verification Features:**
- ✓ Checksum match: "All input files verified (checksum OK)"
- ⚠ Missing files: Lists files that no longer exist
- ⚠ Changed files: Lists files with different content (checksum mismatch)

### 4. Unified Single File

Both CLI and GUI modes are in one file:
- No arguments: launches GUI
- With arguments: runs CLI
- Shared core extraction logic
- Consistent metadata handling

## Usage Examples

### CLI Examples

```bash
# Simple extraction (saves verbs.csv + verbs.json)
python SpaCyVerbExtractor.py input.txt -o verbs.csv

# Load previous settings exactly
python SpaCyVerbExtractor.py --load-metadata verbs.json

# Load settings but use different model
python SpaCyVerbExtractor.py --load-metadata verbs.json --model en_core_web_lg -o output_lg.csv

# Verify input files haven't changed
python SpaCyVerbExtractor.py --load-metadata output_old.json
# Exits with error if input checksums don't match

# View metadata without running
cat verbs.json | python3 -m json.tool
```

### GUI Examples

1. **First run:**
   - Add input files
   - Configure settings
   - Click "Start Extraction"
   - Output files and metadata are saved

2. **Reproduce run:**
   - Click "Load Settings from JSON"
   - Select `verbs.json` from first run
   - GUI shows warnings if input files changed
   - Click "Start Extraction" to recreate

3. **Modify previous run:**
   - Click "Load Settings from JSON"
   - Change any settings in GUI
   - Click "Start Extraction" with new settings
   - New output and metadata are saved

## Implementation Details

### New Functions

**`compute_file_md5(path: Path) -> str`**
- Computes MD5 checksum of a file
- Used for input and output verification

**`save_run_metadata(...)`**
- Saves metadata JSON alongside output file
- Called automatically after extraction completes
- Works for both CLI and GUI modes

**`load_run_metadata(json_path: Path) -> Dict`**
- Loads metadata from JSON file
- Returns dictionary with all settings and checksums

**`load_metadata_from_file(json_path: Path)` (GUI method)**
- GUI-specific metadata loading
- Verifies input file checksums
- Pre-populates all settings
- Shows warnings for issues

### CLI Arguments

**New argument:**
```
--load-metadata FILE    Load settings from previous run JSON file
                       (CLI args override loaded settings)
```

**Changed defaults:**
- `--model`, `--encoding`, `--chunk-size`, `--overlap`, `--dedupe-window`, `--heartbeat-chunks`
- Now default to `None` internally
- Use metadata values if loading, otherwise use hardcoded defaults

### GUI Updates

**New button:**
- "Load Settings from JSON" in Settings section
- Opens file dialog and calls `load_metadata_from_file()`

**New method:**
- `load_metadata_dialog()` - File dialog wrapper
- `load_metadata_from_file()` - Core loading and verification logic

## Verification & Testing

The script has been validated for:
- ✅ Python syntax (py_compile)
- ✅ Imports (all standard + spacy)
- ✅ Type hints (no errors)
- ✅ File I/O (open, write, read)
- ✅ JSON handling (dump, load)
- ✅ Checksum computation (hashlib.md5)

## Benefits

1. **Reproducibility**: Exact run reconstruction with saved metadata
2. **Verification**: Input file checksums detect data changes
3. **Flexibility**: CLI options override loaded settings
4. **User-friendly**: GUI warns about issues and pre-populates fields
5. **Audit trail**: Complete record of extraction settings and statistics
6. **Single file**: Easier maintenance and distribution

## Files Modified

- `SpaCyVerbExtractor.py` - Added metadata support (both CLI and GUI)
- `README.md` - Updated documentation
- `pyproject.toml` - No changes needed (already includes dependencies)

## Backward Compatibility

- ✅ Existing CLI usage without `--load-metadata` works as before
- ✅ Old extractions without JSON metadata still work
- ✅ All default values preserved for non-metadata runs
- ✅ No breaking changes to output format
