# filterSpaCyVerbs: Enhanced with CLI + GUI + Metadata

## Overview

`filterSpaCyVerbs.py` now includes full CLI and GUI support with JSON metadata generation and verification, matching the advanced features of `SpaCyVerbExtractor.py`.

## Usage Modes

### GUI Mode (Default)
Launch without arguments:
```bash
python filterSpaCyVerbs.py
```

Opens an interactive window with:
- Input/output file selection
- Field and frequency threshold controls
- "Load Settings from JSON" button
- Real-time progress logging
- Metadata verification warnings

### CLI Mode
Provide arguments:
```bash
python filterSpaCyVerbs.py input.csv output.csv --field lemma --min-freq 10
python filterSpaCyVerbs.py input.csv output.csv --field lemma --min-freq 10 --max-freq 1000
python filterSpaCyVerbs.py --help
```

## Key Features

### 1. JSON Metadata System

#### Saving Metadata
Every filtering operation generates a `.json` file alongside the output CSV:

```json
{
  "timestamp": "2026-01-24T15:30:45.123456Z",
  "tool": "filterSpaCyVerbs",
  "input_file": "/path/to/verbs.csv",
  "input_checksum": "a1b2c3d4e5f6...",
  "output_file": "/path/to/filtered_verbs.csv",
  "output_checksum": "f6e5d4c3b2a1...",
  "settings": {
    "field": "lemma",
    "min_freq": 10,
    "max_freq": 1000
  },
  "statistics": {
    "rows_written": 245,
    "unique_values": 1523
  }
}
```

#### Loading Metadata
The tool supports loading from two types of metadata files:

**Option A: Previous filterSpaCyVerbs metadata**
```bash
python filterSpaCyVerbs.py --load-metadata filtered_verbs.json input.csv output2.csv
```
- Loads all settings from the JSON
- CLI arguments override loaded settings
- Verifies input file checksum

**Option B: SpaCyVerbExtractor metadata**
GUI or CLI can load the output of `SpaCyVerbExtractor.py`:
```bash
python filterSpaCyVerbs.py --load-metadata verbs_with_mlm.json
```
- Uses the extractor output as input
- Verifies checksum of extracted verbs
- Applies new filtering settings

### 2. Input File Verification

#### CLI Behavior
When loading metadata, checksum is verified automatically:
```
⚠ Input file has changed (checksum mismatch): /path/to/verbs.csv
```
Warning is printed but filtering proceeds (non-blocking).

#### GUI Behavior
When loading metadata, checksum verification occurs:
- If match: ✓ confirmation message
- If mismatch: Warning dialog appears, user chooses to continue or cancel

## Workflow Examples

### Example 1: Filter with Metadata from Extractor

**Step 1:** Extract verbs with SpaCyVerbExtractor
```bash
python SpaCyVerbExtractor.py documents/ verbs_extracted.csv
```
Produces: `verbs_extracted.csv` and `verbs_extracted.json`

**Step 2:** Filter the extracted verbs
```bash
python filterSpaCyVerbs.py --load-metadata verbs_extracted.json \
  verbs_extracted.csv filtered_verbs.csv --field lemma --min-freq 10
```

Output:
- `filtered_verbs.csv` - filtered results
- `filtered_verbs.json` - metadata with checksums

### Example 2: Reproduce Filter Run

**Step 1:** Run filtering operation
```bash
python filterSpaCyVerbs.py verbs.csv output.csv --field lemma --min-freq 20
```

**Step 2:** Later, reproduce exact same filtering
```bash
python filterSpaCyVerbs.py --load-metadata output.json verbs.csv output2.csv
```

The new output will have:
- Same filtering settings
- Checksum verification if input file hasn't changed
- New output and metadata files

### Example 3: Interactive GUI with Metadata

1. Launch GUI: `python filterSpaCyVerbs.py`
2. Click "Load Settings from JSON"
3. Select a filter metadata file
   - Settings auto-populate
   - Checksum verification shown in log
4. Modify settings if needed
5. Click "Start Filtering"
6. Monitor progress in log panel

## API Functions

### Core Filtering
```python
# Count field frequencies (streaming)
counts = count_field_freq(Path("verbs.csv"), "lemma")

# Filter rows with frequency constraints
rows_written = filter_rows(
    input_path=Path("verbs.csv"),
    output_path=Path("filtered.csv"),
    field="lemma",
    counts=counts,
    min_freq=10,
    max_freq=1000
)
```

### Metadata Operations
```python
# Compute MD5 checksum
checksum = compute_file_md5(Path("verbs.csv"))

# Save metadata
save_filter_metadata(
    output_path=Path("filtered.csv"),
    input_path=Path("verbs.csv"),
    input_checksum="a1b2c3...",
    output_checksum="f6e5d4...",
    args={"field": "lemma", "min_freq": 10, "max_freq": 1000},
    stats={"rows_written": 245, "unique_values": 1523}
)

# Load metadata
metadata = load_filter_metadata(Path("filtered.json"))

# Verify checksum
matches, message = verify_input_checksum(Path("verbs.csv"), metadata)
```

## Dependencies

### Core (always required)
- Python 3.9+
- Standard library: `argparse`, `csv`, `hashlib`, `json`, `pathlib`, `datetime`

### GUI (optional, only for GUI mode)
- `PySide6`: Install with `pip install PySide6`

If PySide6 is not installed and you try to launch GUI mode, you'll get:
```
Error: PySide6 is required for GUI mode.
Install it with: pip install PySide6

Or run in CLI mode by providing arguments. Use --help for usage.
```

## File Size and Memory

### Streaming Efficiency
- **Pass 1 (counting):** Streams input CSV, stores only Counter of unique values
- **Pass 2 (filtering):** Streams input, writes output, no full-file buffering
- **Memory usage:** O(unique_values), not O(rows)

Example: 1M row CSV with 10K unique values uses ~10K entries in Counter

### Checksum Computation
- Streams file in 8KB chunks
- O(1) memory regardless of file size

## Error Handling

### CLI Errors
```bash
# Missing required argument
$ python filterSpaCyVerbs.py
Error: input_csv and output_csv are required (or use --load-metadata)

# Invalid field
$ python filterSpaCyVerbs.py verbs.csv out.csv --field invalid
Error: Field 'invalid' not found in input header: ...

# Invalid frequency range
$ python filterSpaCyVerbs.py verbs.csv out.csv --field lemma --min-freq 100 --max-freq 10
Error: --min-freq cannot be greater than --max-freq
```

### GUI Errors
- Displayed in dialog boxes
- Invalid file paths: "Input file not found"
- Missing settings: "Please select an input CSV file"
- Checksum mismatches: Warning with option to proceed or cancel

## Interoperability

### With SpaCyVerbExtractor.py
✓ Accepts output JSON from extractor
✓ Uses extractor's output CSV as input
✓ Verifies checksums across tools

### With Other Tools
- Generates standard CSV output (compatible with pandas, sql, etc.)
- JSON metadata follows same structure as SpaCyVerbExtractor
- MD5 checksums are standard format

## Performance Tips

1. **For large files:** Use CLI mode (no GUI overhead)
   ```bash
   python filterSpaCyVerbs.py large.csv filtered.csv --field lemma --min-freq 5
   ```

2. **For reproducibility:** Always use `--load-metadata`
   ```bash
   python filterSpaCyVerbs.py --load-metadata output.json large.csv output2.csv
   ```

3. **For verification:** Check checksums in JSON before reprocessing
   ```bash
   cat filtered.json | grep -A 2 "checksum"
   ```

## Version History

- **Current:** Full CLI+GUI+Metadata support
- **Previous:** CLI-only, no metadata, no verification
