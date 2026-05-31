# GPU Acceleration on Apple Silicon (M1/M2/M3)

Your setup is correctly configured for GPU acceleration using Metal Performance Shaders (MPS)!

## ✅ Verified Working

- **PyTorch**: 2.10.0 with MPS support
- **spaCy**: 3.7.5
- **Thinc**: 8.2.4 with MPSOps backend
- **MPS Backend**: Active and tested

## Usage

### Option 1: Use the MPS Launcher (Recommended)

The `run_with_mps.py` launcher sets `PYTORCH_ENABLE_MPS_FALLBACK=1` BEFORE PyTorch is imported, which is critical for proper MPS operation:

```bash
# Run TextVerbGroupCounter with GPU acceleration
uv run python run_with_mps.py input.csv groups.csv output.xlsx --text-col text --model en_core_web_trf

# With other options
uv run python run_with_mps.py input.csv groups.csv output.xlsx \
  --text-col text \
  --model en_core_web_trf \
  --batch-size 32 \
  --include-aux
```

### Option 2: Set Environment Variable Manually

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run TextVerbGroupCounter.py input.csv groups.csv output.xlsx --text-col text
```

### Option 3: Force CPU (if you encounter issues)

```bash
uv run TextVerbGroupCounter.py input.csv groups.csv output.xlsx --text-col text --force-cpu
```

## Expected Performance

Based on [Explosion AI's benchmarks](https://explosion.ai/blog/metal-performance-shaders), you should see:

| Device | Speedup vs CPU |
|--------|----------------|
| M1 (8 GPU cores) | 1.9x |
| M2 (10 GPU cores) | 2.7x |
| M1 Pro (14 GPU cores) | 2.9x |
| M1 Max (32 GPU cores) | 4.7x |
| M1 Ultra (48 GPU cores) | 5.5x |

## How It Works

1. **MPS Fallback**: The `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable tells PyTorch to:
   - Use MPS (Metal) for operations that ARE supported (matrix multiplication, convolutions, etc.)
   - Automatically fall back to CPU for operations that AREN'T supported
   - This prevents "Placeholder storage has not been allocated on MPS device" errors

2. **Thinc MPSOps**: spaCy's underlying library (Thinc) detects MPS and uses the `MPSOps` backend, which places PyTorch transformer layers on the `mps` device.

3. **Transparent Acceleration**: You don't need to change your code - operations are automatically accelerated when possible.

## Verifying GPU Usage

Run the test script to verify MPS is active:

```bash
uv run python test_mps.py
```

You should see:
```
✓ Matrix multiplication on MPS: SUCCESS
✓ Embedding layer on MPS: SUCCESS
✓ spaCy GPU enabled (GPU True)
✓ Thinc backend: MPSOps
🎉 SUCCESS! MPS acceleration is active!
```

## Troubleshooting

### Still getting "Placeholder storage" errors?

This usually means `PYTORCH_ENABLE_MPS_FALLBACK` wasn't set before PyTorch was imported. Solutions:

1. **Use the launcher**: `uv run python run_with_mps.py <args>` (sets env var first)
2. **Set in shell**: `export PYTORCH_ENABLE_MPS_FALLBACK=1` before running
3. **Force CPU**: Add `--force-cpu` flag to disable GPU entirely

### Want to verify which operations use MPS vs CPU?

Set PyTorch logging:
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Force MPS to log fallback operations
uv run python run_with_mps.py <args>
```

### Performance seems slow?

- Check batch size: `--batch-size 32` (default) works well for most systems
- Larger batches (64-128) may help on M1 Max/Ultra
- Smaller batches (16) if running out of memory

## Technical Details

- **Article**: https://explosion.ai/blog/metal-performance-shaders
- **Required versions**:
  - PyTorch ≥ 1.13.0 (for MPS support)
  - spaCy ≥ 3.4.2 (for MPS coordination)
  - Thinc ≥ 8.1.0 (for MPSOps backend)
  - spacy-transformers ≥ 1.1.8 (for transformer + MPS)
- **Current versions**: All requirements exceeded ✓

## Summary

✅ **GPU acceleration is ready to use!**

Just run:
```bash
uv run python run_with_mps.py input.csv groups.csv output.xlsx --text-col text --model en_core_web_trf
```

Expect 2-5x speedup depending on your Mac's GPU cores.
