# Full Logging Setup

Your TextVerbGroupCounter now logs all events to both console and file.

## Log Output

All logs are automatically saved to: **`TextVerbGroupCounter.log`**

### Log Format

- **File**: Detailed logs with timestamps (`2026-02-07 14:23:45 - INFO: ...`)
- **Console**: Brief output (`INFO: ...`)

## What's Logged

✅ **Startup**
- Device detection (CPU/GPU/Metal)
- MPS status
- Batch size configuration

✅ **Processing**
- Batch number and size
- Starting row for each batch
- Full error tracebacks if processing fails

✅ **Per-Batch**
- Row ranges being processed
- Errors with complete stack traces
- Batch completion status

## To Debug Your Issue

### Run normally
```bash
uv run python run_with_mps.py  # GUI mode
# or with arguments for CLI mode
uv run python run_with_mps.py input.csv groups.csv output.xlsx --text-col text
```

### View the log
```bash
# View full log
python view_log.py

# Follow log in real-time (like tail -f)
python view_log.py --follow
```

### Find the first batch error

The log will show:
```
2026-02-07 14:23:45 - INFO: Starting row processing with batch_size=32
2026-02-07 14:23:46 - INFO: Processing batch 1 with 32 rows (starting at row 1)
...
2026-02-07 14:24:12 - ERROR: Error processing batch of 32 rows at row <N>: Placeholder storage has not been allocated on MPS device!
```

This will tell us exactly which batch (and row number) fails first.

## Example Log Output

```
2026-02-07 14:23:45 - INFO: === TextVerbGroupCounter Starting ===
2026-02-07 14:23:45 - INFO: Device: Metal (Apple Silicon)
2026-02-07 14:23:45 - INFO: Details: MPS available and built
2026-02-07 14:23:46 - INFO: Loading spaCy model: en_core_web_trf
2026-02-07 14:23:46 - INFO: GPU acceleration enabled for spaCy (GPU True)
2026-02-07 14:23:46 - INFO: Starting row processing with batch_size=32
Processing rows...
2026-02-07 14:23:47 - INFO: Processing batch 1 with 32 rows (starting at row 1)
2026-02-07 14:23:48 - INFO: Processing batch 2 with 32 rows (starting at row 33)
...
2026-02-07 14:25:12 - ERROR: Error processing batch of 32 rows at row 8672: Placeholder storage has not been allocated on MPS device!
2026-02-07 14:25:12 - ERROR: Traceback (most recent call last):
  ...
```

## For Your Bug Report

After running the script, share:
1. The full `TextVerbGroupCounter.log` file
2. This will show the exact batch number and row where it first fails
3. Include full timestamps and error context
