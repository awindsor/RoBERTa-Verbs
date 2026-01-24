# Checksum Verification Update

## Summary

Both CLI and GUI now warn if input files match the name in the JSON metadata but have different checksums (indicating file modification).

## Changes Made

### 1. New Function: `verify_input_file_checksums()`
Located at line 104 in SpaCyVerbExtractor.py

```python
def verify_input_file_checksums(input_paths: List[Path], metadata: Dict) -> Dict[str, str]:
    """
    Verify input file checksums against metadata.
    
    Returns dict with issues found:
    {
        "missing_files": [list of paths that don't exist],
        "checksum_mismatches": [list of paths with different checksums],
    }
    """
```

Compares provided input files against checksums stored in metadata:
- Returns list of files that are missing
- Returns list of files that have changed (checksum mismatch)

### 2. CLI Updates
Location: After `iter_paths()` call (~line 287)

**Added verification block:**
```python
if args.load_metadata:
    logger.info("Verifying input file checksums against metadata...")
    issues = verify_input_file_checksums(paths, metadata)
    
    if issues["missing_files"]:
        logger.warning("⚠ Missing input files:")
        for f in issues["missing_files"]:
            logger.warning(f"   {f}")
    
    if issues["checksum_mismatches"]:
        logger.warning("⚠ Input files have changed (checksum mismatch):")
        for f in issues["checksum_mismatches"]:
            logger.warning(f"   {f}")
    
    if issues["missing_files"] or issues["checksum_mismatches"]:
        logger.warning("⚠ Proceeding with extraction despite file changes")
    else:
        logger.info("✓ All input files verified (checksum OK)")
```

**Behavior:**
- Runs automatically when `--load-metadata` is used
- Logs warnings for each issue found
- Continues with extraction (doesn't block)
- Shows success message if all files match

### 3. GUI Updates
Location: `start_extraction()` method (~line 1038)

**Added verification block:**
```python
# Check if any input files match metadata (if metadata was loaded)
metadata_path = output_path.with_suffix(".json")
if metadata_path.exists():
    try:
        metadata = load_run_metadata(metadata_path)
        issues = verify_input_file_checksums(self.input_files, metadata)
        
        if issues["missing_files"] or issues["checksum_mismatches"]:
            warnings = []
            if issues["missing_files"]:
                warnings.append(f"⚠ Missing from metadata:\n" + "\n".join(issues["missing_files"]))
            if issues["checksum_mismatches"]:
                warnings.append(f"⚠ Changed since last run (checksum mismatch):\n" + "\n".join(issues["checksum_mismatches"]))
            
            msg = "\n\n".join(warnings) + "\n\nContinue with extraction?"
            reply = QMessageBox.warning(self, "Input File Checksum Mismatch", msg, QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
    except Exception as e:
        # Metadata issues shouldn't block extraction
        pass
```

**Behavior:**
- Runs before every extraction in GUI
- Checks for metadata file matching output filename
- Shows warning dialog if files have changed
- User can choose to continue or cancel
- Errors in checking metadata don't block extraction

## Usage Examples

### CLI: Running with changed input files

```bash
python SpaCyVerbExtractor.py --load-metadata old_output.json
```

Output:
```
INFO: Verifying input file checksums against metadata...
WARNING: ⚠ Input files have changed (checksum mismatch):
WARNING:    /path/file1.txt
WARNING: ⚠ Proceeding with extraction despite file changes
```

### CLI: All files match

```bash
python SpaCyVerbExtractor.py --load-metadata old_output.json
```

Output:
```
INFO: Verifying input file checksums against metadata...
INFO: ✓ All input files verified (checksum OK)
```

### GUI: Running with changed files

1. Add input files (e.g., `modified_file.txt`)
2. Click "Start Extraction"
3. If `modified_file.txt` exists in previous metadata with different checksum:
   - Warning dialog appears
   - Shows "⚠ Changed since last run (checksum mismatch): /path/file.txt"
   - User clicks "Yes" to continue or "No" to cancel

### GUI: Using previously loaded metadata

1. Click "Load Settings from JSON"
2. Select `output.json` from previous run
3. Input files pre-filled (if available)
4. Click "Start Extraction"
5. If any files changed: warning appears before extraction

## Edge Cases Handled

1. **Metadata file missing**: No verification runs (graceful skip)
2. **Input file doesn't exist**: Listed as "missing"
3. **Input file unchanged**: No warning (checksum matches)
4. **Metadata loading error**: Doesn't block extraction (try-except)
5. **No metadata loaded**: Verification skipped (CLI only runs with `--load-metadata`)

## Verification Flow

```
CLI/GUI START
    ↓
[GUI only] Check if output.json exists
    ↓
Load metadata from JSON (if exists/loaded)
    ↓
Compare input file checksums
    ↓
Issues found?
    ├─ Yes → Show warning
    │   ├─ CLI: Log warning, continue
    │   └─ GUI: Show dialog, ask user
    └─ No → Proceed silently
    ↓
Start extraction
```

## Technical Details

- **Checksum algorithm**: MD5 (same as used elsewhere)
- **Performance**: Minimal overhead (only on direct input file verification)
- **Backward compatible**: Works with or without metadata
- **Cross-platform**: Works on Windows, macOS, Linux
