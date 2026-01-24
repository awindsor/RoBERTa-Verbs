# SpaCyVerbExtractor Quick Reference

## Launch Modes

```bash
# GUI mode (interactive)
python SpaCyVerbExtractor.py

# CLI mode (with arguments)
python SpaCyVerbExtractor.py input.txt -o output.csv
python SpaCyVerbExtractor.py --help
```

## Common CLI Usage

### Basic Extraction
```bash
python SpaCyVerbExtractor.py input.txt -o verbs.csv
# Creates: verbs.csv + verbs.json (metadata)
```

### Multiple Files
```bash
python SpaCyVerbExtractor.py file1.txt file2.txt file3.txt -o verbs.csv
```

### Using Paths File
```bash
python SpaCyVerbExtractor.py --paths-file files.txt -o verbs.csv
# files.txt format: one path per line, # for comments
```

### Output Options
```bash
python SpaCyVerbExtractor.py input.txt -o verbs.tsv --tsv
# TSV output instead of CSV
```

### Include Auxiliary Verbs
```bash
python SpaCyVerbExtractor.py input.txt -o verbs.csv --include-aux
```

### Performance Tuning
```bash
python SpaCyVerbExtractor.py input.txt -o verbs.csv \
  --chunk-size 5000000 \
  --overlap 10000 \
  --dedupe-window 100000
```

### Model Selection
```bash
python SpaCyVerbExtractor.py input.txt -o verbs.csv --model en_core_web_lg
```

## Metadata Features

### Load Previous Run Settings
```bash
# Exact reproduction
python SpaCyVerbExtractor.py --load-metadata verbs.json

# With overrides
python SpaCyVerbExtractor.py --load-metadata verbs.json --chunk-size 1000000 -o new_output.csv

# Verify input files haven't changed
python SpaCyVerbExtractor.py --load-metadata verbs.json
# Fails if input checksums don't match
```

### View Metadata
```bash
# Pretty-print JSON
cat verbs.json | python3 -m json.tool

# Check specific field
python3 -c "import json; m=json.load(open('verbs.json')); print(m['statistics'])"
```

## GUI Usage

### First Run
1. Click "Add Files" or "Add Paths File"
2. Set output filename
3. (Optional) Configure settings
4. Click "Start Extraction"
5. Monitor progress in log

### Reproduce Run
1. Click "Load Settings from JSON"
2. Select previous `.json` file
3. Review warnings (if any)
4. (Optional) Modify any settings
5. Click "Start Extraction"

### Verify Files
- Checksums shown in log if mismatches found
- ✓ All input files verified (checksum OK)
- ⚠ Input files have changed (checksum mismatch)
- ⚠ Missing input files

## Output Files

```
extraction_output/
├── verbs.csv                 # Main output (or .tsv)
├── verbs.json               # Metadata file
├── settings                 # All extraction settings
├── input_checksums          # MD5 checksums of input files
├── output_checksum          # MD5 of output file
└── statistics               # Extraction stats
```

## CLI Arguments Reference

```
positional arguments:
  paths                 Paths to text files

optional arguments:
  --paths-file FILE          Text file with one path per line
  -o, --output FILE          Output file path (default: verbs.csv)
  --load-metadata FILE       Load settings from previous run JSON
  --tsv                      Write TSV instead of CSV
  --model MODEL              spaCy model name (default: en_core_web_sm)
  --encoding ENC             Input file encoding (default: utf-8)
  --include-aux              Also treat AUX tokens as verbs
  --chunk-size NUM           Chunk size in chars (default: 2,000,000)
  --overlap NUM              Chunk overlap in chars (default: 5,000)
  --dedupe-window NUM        De-dup cache size (default: 50,000)
  --heartbeat-chunks NUM     Progress log interval (default: 10)
  --log-level LEVEL          Logging verbosity (DEBUG|INFO|WARNING|ERROR)
```

## Troubleshooting

### "PySide6 is required for GUI mode"
```bash
pip install PySide6
```

### "Input files have changed (checksum mismatch)"
- Files were modified since last extraction
- To proceed, either:
  - Restore original files from backup
  - Run without `--load-metadata` flag
  - Accept the warning in GUI and proceed

### "Missing input files"
- Files were moved or deleted
- To proceed, either:
  - Restore files to original location
  - Provide different input files with `-o` in CLI
  - Add files in GUI and run

### Model not found
```bash
python3 -m spacy download en_core_web_sm
python3 -m spacy download en_core_web_md
python3 -m spacy download en_core_web_lg
```

## Performance Tips

### Large Files
- Increase `--chunk-size`: 5,000,000-10,000,000 characters
- Decrease `--heartbeat-chunks` for less logging overhead
- Use `--overlap 0` if sentence boundary loss is acceptable

### Memory Constraints
- Decrease `--chunk-size`: 500,000-1,000,000 characters
- Decrease `--dedupe-window` for smaller de-dup cache

### Speed
- Use smaller model: `--model en_core_web_sm`
- Decrease `--overlap` (trade-off: may miss some sentences)
- Increase `--chunk-size` (trade-off: more memory)
