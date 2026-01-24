# filterSpaCyVerbs + SpaCyVerbExtractor Integration

## Overview

`filterSpaCyVerbs.py` now fully integrates with `SpaCyVerbExtractor.py`, allowing you to:
- Load extractor outputs directly
- Verify checksums across tools
- Reconstruct filtering workflows
- Create reproducible analysis pipelines

## Two Types of Metadata

### Type 1: SpaCyVerbExtractor Metadata
**File:** `verbs_extracted.json`

Contains information about verb extraction:
```json
{
  "timestamp": "2026-01-24T14:00:00Z",
  "tool": "SpaCyVerbExtractor",
  "input_file": "documents/",
  "output_file": "verbs_extracted.csv",
  "output_checksum": "abc123...",
  "settings": {
    "spacy_model": "en_core_web_sm",
    "lemmatize": true
  },
  "statistics": {
    "documents_processed": 1000,
    "total_tokens": 50000,
    "verbs_found": 2500
  }
}
```

### Type 2: filterSpaCyVerbs Metadata
**File:** `filtered_verbs.json`

Contains information about filtering:
```json
{
  "timestamp": "2026-01-24T15:00:00Z",
  "tool": "filterSpaCyVerbs",
  "input_file": "verbs_extracted.csv",
  "input_checksum": "abc123...",
  "output_file": "filtered_verbs.csv",
  "output_checksum": "def456...",
  "settings": {
    "field": "lemma",
    "min_freq": 10,
    "max_freq": null
  },
  "statistics": {
    "rows_written": 245,
    "unique_values": 1523
  }
}
```

## Complete Workflow: Extraction → Filtering

### Step-by-Step

**Step 1: Extract verbs using SpaCyVerbExtractor**
```bash
cd /path/to/documents
python SpaCyVerbExtractor.py . extracted_verbs.csv --batch-size 5
```

Output:
```
extracted_verbs.csv        (all extracted verbs)
extracted_verbs.json       (extraction metadata)
```

**Step 2: Load extractor output in filterSpaCyVerbs (CLI)**
```bash
python filterSpaCyVerbs.py --load-metadata extracted_verbs.json \
  extracted_verbs.csv filtered_verbs.csv --field lemma --min-freq 10
```

The tool:
1. Reads metadata from `extracted_verbs.json`
2. Uses `extracted_verbs.csv` as input (from metadata)
3. Applies new filtering settings
4. Generates `filtered_verbs.csv` and `filtered_verbs.json`

**Step 3: Load extractor output in filterSpaCyVerbs (GUI)**
1. Run: `python filterSpaCyVerbs.py`
2. Click: "Load Settings from JSON"
3. Select: `extracted_verbs.json`
4. Result: Input path auto-populated with extracted verbs file
5. Set: Frequency thresholds as desired
6. Click: "Start Filtering"

## Checksum Verification Chain

```
SpaCyVerbExtractor
        ↓
  verbs_extracted.csv + extracted_verbs.json
        ↓
filterSpaCyVerbs (load metadata)
        ↓
  [VERIFY extracted_verbs.csv checksum]
        ↓
  filtered_verbs.csv + filtered_verbs.json
        ↓
  (filtered metadata contains input checksum)
```

### Example Verification Sequence

**Run 1: Initial extraction and filtering**
```bash
# Extract
python SpaCyVerbExtractor.py documents/ verbs.csv

# Filter - loads extractor output
python filterSpaCyVerbs.py --load-metadata verbs.json verbs.csv filtered.csv --field lemma --min-freq 10
```

**verbs.json contents:**
```json
{
  "tool": "SpaCyVerbExtractor",
  "output_file": "verbs.csv",
  "output_checksum": "a1b2c3d4e5f6"
}
```

**filtered.json contents:**
```json
{
  "tool": "filterSpaCyVerbs",
  "input_file": "verbs.csv",
  "input_checksum": "a1b2c3d4e5f6",
  "output_file": "filtered.csv",
  "output_checksum": "f6e5d4c3b2a1"
}
```

**Run 2: Later, reproduce the filtering**
```bash
# If verbs.csv hasn't changed (same checksum):
python filterSpaCyVerbs.py --load-metadata filtered.json verbs.csv filtered2.csv

# Output:
# ✓ Input file verified (checksum OK)
# ✓ Wrote 245 rows to filtered2.csv
# ✓ Saved metadata to filtered2.json
```

**Run 3: If verbs.csv was modified**
```bash
# If verbs.csv was changed (different checksum):
python filterSpaCyVerbs.py --load-metadata filtered.json verbs.csv filtered2.csv

# Output:
# ⚠ Input file has changed (checksum mismatch): verbs.csv
# (filtering proceeds anyway, but warning is shown)
# ✓ Wrote 245 rows to filtered2.csv
```

## Smart Metadata Loading

`filterSpaCyVerbs.py` automatically detects metadata type:

```python
metadata = load_filter_metadata(Path("some.json"))
tool = metadata.get("tool")

if tool == "filterSpaCyVerbs":
    # Load as filter settings
    settings = metadata["settings"]
    input_file = metadata["input_file"]
    
elif tool == "SpaCyVerbExtractor":
    # Load as input data source
    input_file = metadata["output_file"]
    verify_checksum(input_file, metadata)
```

This means:
- ✓ Load `extracted_verbs.json` → uses `extracted_verbs.csv`
- ✓ Load `filtered_verbs.json` → uses settings from previous filter
- ✓ Either can be loaded in GUI or CLI

## Use Cases

### Use Case 1: Parameter Sweep
Extract once, filter with multiple parameter sets:

```bash
# Extract
python SpaCyVerbExtractor.py documents/ verbs.csv

# Try different frequency thresholds
python filterSpaCyVerbs.py --load-metadata verbs.json verbs.csv filtered_10.csv --field lemma --min-freq 10
python filterSpaCyVerbs.py --load-metadata verbs.json verbs.csv filtered_20.csv --field lemma --min-freq 20
python filterSpaCyVerbs.py --load-metadata verbs.json verbs.csv filtered_50.csv --field lemma --min-freq 50
```

Each produces:
- `filtered_10.csv`, `filtered_10.json`
- `filtered_20.csv`, `filtered_20.json`
- `filtered_50.csv`, `filtered_50.json`

All have verified checksums of input.

### Use Case 2: Reproducible Research
Document exact pipeline:

```bash
# In your research notebook:
python SpaCyVerbExtractor.py corpus/ verbs.csv
python filterSpaCyVerbs.py --load-metadata verbs.json verbs.csv filtered.csv --field lemma --min-freq 15

# Years later, reproduce exactly:
# Check that verbs.csv matches original
cat filtered.json | grep "input_checksum"

# Reprocess with same settings
python filterSpaCyVerbs.py --load-metadata filtered.json verbs.csv filtered_v2.csv
```

### Use Case 3: Data Integrity Checks
Verify entire pipeline before downstream analysis:

```bash
# Check extraction
python -c "import json; m = json.load(open('verbs.json')); print(f\"Extracted {m['statistics']['verbs_found']} verbs\")"

# Check filtering
python -c "import json; m = json.load(open('filtered.json')); print(f\"Filtered to {m['statistics']['rows_written']} rows\")"

# Verify no data loss due to file corruption
python -c "
import json
with open('verbs.json') as f:
    extraction = json.load(f)
with open('filtered.json') as f:
    filtering = json.load(f)
if extraction['output_checksum'] == filtering['input_checksum']:
    print('✓ Data integrity verified')
else:
    print('✗ Data mismatch detected!')
"
```

## Common Patterns

### Pattern 1: Extract → Filter → Analyze

```bash
# Extract (once)
python SpaCyVerbExtractor.py documents/ verbs.csv

# Filter (iteratively)
python filterSpaCyVerbs.py --load-metadata verbs.json verbs.csv filtered.csv --field lemma --min-freq 10

# Analyze (your custom script)
python analyze.py filtered.csv
```

### Pattern 2: Load Previous Filter, Apply New Settings

```bash
# From GUI:
1. "Load Settings from JSON" → select filtered.json
2. Fields and frequencies auto-populate
3. Modify frequency thresholds as needed
4. Click "Start Filtering"
```

```bash
# From CLI:
python filterSpaCyVerbs.py --load-metadata filtered.json verbs.csv filtered_new.csv --field lemma --min-freq 20
```

### Pattern 3: Audit Trail

All operations leave JSON traces:

```bash
# See extraction settings
cat verbs.json | jq '.settings'

# See filtering settings
cat filtered.json | jq '.settings'

# See row counts
cat filtered.json | jq '.statistics.rows_written'

# See checksums for verification
cat filtered.json | jq '.input_checksum, .output_checksum'
```

## Troubleshooting

### Problem: "Input file has changed (checksum mismatch)"
**Cause:** File was modified since metadata was created
**Solutions:**
1. Restore original file from backup
2. Regenerate metadata with new file
3. Accept warning and proceed (non-blocking)

### Problem: "Unknown metadata format"
**Cause:** JSON is not from SpaCyVerbExtractor or filterSpaCyVerbs
**Solution:** Check JSON file type: `cat file.json | jq '.tool'`

### Problem: GUI doesn't load file paths
**Cause:** Path in metadata is absolute and no longer valid
**Solution:** Click "Browse" to select file, then re-save path

## JSON Structure Reference

### SpaCyVerbExtractor JSON Keys
```
tool
timestamp
input_file
output_file
output_checksum
settings (spacy_model, lemmatize, min_confidence, etc.)
statistics (documents_processed, verbs_found, etc.)
```

### filterSpaCyVerbs JSON Keys
```
tool
timestamp
input_file
input_checksum          ← Always present when saving
output_file
output_checksum         ← Always present when saving
settings (field, min_freq, max_freq)
statistics (rows_written, unique_values)
```

The `input_checksum` is the key that enables verification of filter reproducibility.
