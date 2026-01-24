# filterSpaCyVerbs: Quick Reference

## Launch Modes

```bash
# GUI mode (interactive)
python filterSpaCyVerbs.py

# CLI mode (command-line)
python filterSpaCyVerbs.py input.csv output.csv --field lemma --min-freq 10
python filterSpaCyVerbs.py --help
```

## Basic CLI Examples

### Filter by Lemma (Min Frequency Only)
```bash
python filterSpaCyVerbs.py verbs.csv filtered.csv --field lemma --min-freq 10
```
- Keeps lemmas that appear 10+ times
- Generates `filtered.csv` and `filtered.csv.json`

### Filter by Surface Form with Range
```bash
python filterSpaCyVerbs.py verbs.csv filtered.csv --field surface_lower --min-freq 5 --max-freq 100
```
- Keeps surface forms with frequency between 5-100
- Discards very rare (< 5) and very common (> 100) forms

### Filter with Max Only
```bash
python filterSpaCyVerbs.py verbs.csv filtered.csv --field lemma --max-freq 1000
```
- Keeps lemmas appearing ≤ 1000 times
- Useful for removing overly frequent common verbs

## Metadata Workflows

### Load Previous Filter Settings
```bash
python filterSpaCyVerbs.py --load-metadata output.json verbs.csv filtered2.csv
```
- Loads all settings from `output.json`
- CLI args override loaded settings
- Verifies input checksum

### Load SpaCyVerbExtractor Output
```bash
python filterSpaCyVerbs.py --load-metadata extracted_verbs.json extracted_verbs.csv filtered.csv --field lemma --min-freq 10
```
- Uses extractor's output as input
- Verifies extractor output hasn't changed
- Applies new filtering settings
- Generates new metadata

### Reproduce Exact Filter Run
```bash
# Create metadata file by running filtering once
python filterSpaCyVerbs.py verbs.csv output.csv --field lemma --min-freq 20
# Now output.json contains the exact settings

# Later, reproduce (verifies input file unchanged)
python filterSpaCyVerbs.py --load-metadata output.json verbs.csv output2.csv
```

## Output Files

Each run generates two files:

**CSV Output**
```
filtered_verbs.csv
- Standard CSV format
- Compatible with pandas, SQL, etc.
- Preserves all original columns
```

**JSON Metadata**
```
filtered_verbs.json
- Filtering settings
- MD5 checksums (input & output)
- Row counts and unique values
- Timestamp
```

View metadata:
```bash
cat filtered_verbs.json | jq '.'              # Pretty print
cat filtered_verbs.json | jq '.settings'      # Just settings
cat filtered_verbs.json | jq '.statistics'    # Just stats
```

## Common Workflows

### Workflow 1: Extract → Filter → Analyze
```bash
# Step 1: Extract all verbs from documents
python SpaCyVerbExtractor.py documents/ verbs.csv

# Step 2: Filter to high-frequency verbs
python filterSpaCyVerbs.py --load-metadata verbs.json verbs.csv filtered.csv --field lemma --min-freq 10

# Step 3: Analyze (your script)
python my_analysis.py filtered.csv
```

### Workflow 2: Parameter Sweep
```bash
# Extract once
python SpaCyVerbExtractor.py documents/ verbs.csv

# Try multiple frequency thresholds
for min_freq in 5 10 20 50; do
  python filterSpaCyVerbs.py --load-metadata verbs.json verbs.csv filtered_$min_freq.csv --field lemma --min-freq $min_freq
done

# Outputs: filtered_5.csv, filtered_10.csv, filtered_20.csv, filtered_50.csv
# Each has verified checksums
```

### Workflow 3: Filter by Different Fields
```bash
# Extract once
python SpaCyVerbExtractor.py documents/ verbs.csv

# Filter lemmas
python filterSpaCyVerbs.py --load-metadata verbs.json verbs.csv filtered_lemma.csv --field lemma --min-freq 10

# Filter surface forms
python filterSpaCyVerbs.py --load-metadata verbs.json verbs.csv filtered_surface.csv --field surface_lower --min-freq 10
```

## Error Messages

```
Error: input_csv and output_csv are required
→ Solution: python filterSpaCyVerbs.py input.csv output.csv [options]

Error: --field is required
→ Solution: add --field lemma or --field surface_lower

Error: At least one of --min-freq or --max-freq must be specified
→ Solution: add --min-freq 10 or --max-freq 1000 or both

Error: --min-freq must be >= 1
→ Solution: use positive integers only

Error: Metadata file not found
→ Solution: check --load-metadata path
```

## Tips & Tricks

### Tip 1: Quick Verification
```bash
# Check what settings were used
python -c "import json; print(json.load(open('filtered.json'))['settings'])"
```

### Tip 2: Find Files with Same Output
```bash
# Check if filtered files have same input data
python -c "
import json
m1 = json.load(open('filtered_v1.json'))
m2 = json.load(open('filtered_v2.json'))
if m1['input_checksum'] == m2['input_checksum']:
    print('Same input data')
else:
    print('Different input data')
"
```

### Tip 3: Dry Run (Lemma Check)
```bash
# Count unique values without filtering
python -c "
import csv
from collections import Counter
with open('verbs.csv') as f:
    reader = csv.DictReader(f)
    lemmas = Counter(row['lemma'] for row in reader)
    print(f'Unique lemmas: {len(lemmas)}')
    print(f'Most common: {lemmas.most_common(5)}')
"
```

### Tip 4: Compare Filter Outputs
```bash
# Compare two filtered results
python -c "
import csv
def count_rows(path):
    with open(path) as f:
        return sum(1 for _ in f) - 1  # exclude header
print('filtered_10.csv:', count_rows('filtered_10.csv'), 'rows')
print('filtered_20.csv:', count_rows('filtered_20.csv'), 'rows')
"
```

## Advanced: Loading Both Types of Metadata

### Load FilterSpaCyVerbs Metadata
```bash
python filterSpaCyVerbs.py --load-metadata filter.json input.csv output.csv
# → Loads settings: field, min_freq, max_freq
# → Verifies input.csv against input_checksum
```

### Load SpaCyVerbExtractor Metadata
```bash
python filterSpaCyVerbs.py --load-metadata extracted.json extracted.csv output.csv --field lemma --min-freq 10
# → Uses extracted.csv as input (from metadata)
# → Verifies extracted.csv against output_checksum
# → Applies new filter settings
```

Both are supported automatically!

## Performance Notes

- **Memory:** O(unique_values), not O(rows)
  - 1M rows, 10K unique values = ~10K entries in memory
- **Time:** Linear in file size
  - 1M row CSV: ~5-10 seconds on modern hardware
- **Checksums:** Streamed, O(1) memory
  - File size doesn't affect memory usage

## Dependencies

**Required:**
- Python 3.9+
- `csv`, `json`, `pathlib`, `hashlib` (standard library)

**Optional (GUI only):**
- `PySide6` (install with `pip install PySide6`)

## See Also

- `FILTERSPACY_FEATURES.md` - Detailed feature documentation
- `FILTERSPACY_EXTRACTOR_INTEGRATION.md` - Integration with SpaCyVerbExtractor
- `SpaCyVerbExtractor.py` - Verb extraction tool
