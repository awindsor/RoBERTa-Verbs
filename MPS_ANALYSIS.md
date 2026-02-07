# Metal Performance Shaders (MPS) Analysis

**Based on:** https://explosion.ai/blog/metal-performance-shaders (Nov 24, 2022)

## Key Findings from Article

### Critical Version Requirements
The article explicitly states MPS support requires:
- **Thinc ≥ 8.1.0** (for `MPSOps` class support)
- **spaCy ≥ 3.4.2** (for MPS coordination)
- **spacy-transformers ≥ 1.1.8** (for transformer + MPS support)
- **PyTorch ≥ 1.13.0** (for MPS device support)

### Recommended Approach
The article provides official code for enabling MPS:
```python
import spacy
spacy.require_gpu()  # Use require_gpu() instead of prefer_gpu()
nlp = spacy.load('de_dep_news_trf')
```

### Verifying GPU Usage
The article shows how to verify MPS is active:
```python
from thinc.api import get_current_ops
ops = get_current_ops()
# Should return: <thinc.backends.mps_ops.MPSOps object at 0x...>
```

### Hardware Details
- **MPS fallback enabled via:** `PYTORCH_ENABLE_MPS_FALLBACK=1` (PyTorch env var)
- **GPU disabled via:** `SPACY_DISABLE_GPU=1` (spaCy env var) - what we're using
- PyTorch AMX (matrix multiply acceleration) is automatic via Apple's Accelerate framework
- GPU provides 2-5.5x speedup on transformer models vs CPU

### Warning Flags
The article emphasizes:
> "Some operations have not been implemented yet... In such cases PyTorch will fall back to CPU kernels when the environment variable `PYTORCH_ENABLE_MPS_FALLBACK` is set to `1`"

This is NOT the same as disabling GPU - it allows GPU to try ops and fall back to CPU on unsupported ones.

---

## Our Current Implementation vs. Recommendations

### ✅ Correct Elements
| Aspect | Our Code | Status |
|--------|----------|--------|
| spaCy version | `>=3.7.0` | ✅ Meets 3.4.2 requirement |
| PyTorch version | `>=2.0.0` | ✅ Exceeds 1.13.0 requirement |
| Disabling GPU option | `SPACY_DISABLE_GPU=1` | ✅ Correct env var |
| Detecting MPS errors | RuntimeError catch | ✅ Appropriate |

### ⚠️ Potential Issues

#### 1. **Missing Thinc Version Requirement**
**Issue:** Thinc is not explicitly listed in dependencies
- Thinc is installed as transitive dependency of spaCy
- Article requires **Thinc ≥ 8.1.0** for MPS support (adds `MPSOps` class)
- We don't verify Thinc version
- spaCy 3.7.0 includes Thinc 8.1.2+, so this is OK but implicit

**Recommendation:** Explicitly add `thinc>=8.1.0` to dependencies for clarity

#### 2. **Missing spacy-transformers Version Pin**
**Issue:** If using transformer models, need `spacy-transformers>=1.1.8`
- Article explicitly recommends this version
- We have transformer-model optional deps but no version lock in main
- Only in dev-optional: `spacy-curated-transformers>=0.2.0,<0.3.0`

**Recommendation:** Pin `spacy-transformers>=1.1.8` in optional deps

#### 3. **Wrong Method: `prefer_gpu()` vs `require_gpu()`**
**Issue:** Article shows official pattern uses `require_gpu()` not `prefer_gpu()`
```python
# Article's recommendation:
spacy.require_gpu()  # ← Stronger guarantee

# Our code:
from spacy import prefer_gpu
if prefer_gpu():  # ← Just a preference
    logger.info("GPU acceleration enabled")
```

- `prefer_gpu()`: Attempts GPU, returns bool, allows fallback
- `require_gpu()`: Asserts GPU available, raises error if not

**Current behavior:** Our try/except on `prefer_gpu()` is reasonable, but not the official pattern

#### 4. **Missing MPS Fallback Environment Variable**
**Issue:** Should set `PYTORCH_ENABLE_MPS_FALLBACK=1` not just disable GPU
- Article shows PyTorch has built-in MPS fallback mechanism
- We use `SPACY_DISABLE_GPU=1` to skip GPU entirely
- Better approach: Let PyTorch try MPS, fall back gracefully

**Current status:** Our approach (full CPU) is conservative but safe. PyTorch fallback would be optimal.

#### 5. **Not Verifying `MPSOps` is Active**
**Issue:** No way to confirm MPS is actually being used
- Article shows checking: `from thinc.api import get_current_ops`
- We only check PyTorch device availability
- No verification that Thinc chose `MPSOps` backend

**Recommendation:** Add optional diagnostic logging with Thinc ops check

#### 6. **Disabled For All Transformers on M1, Not Just Problematic Ones**
**Issue:** Our code disables MPS for ANY transformer model on Apple Silicon
```python
if "trf" in model_name.lower() and gpu_info.get("device") == "Metal (Apple Silicon)":
    os.environ["SPACY_DISABLE_GPU"] = "1"  # ← Too broad
```

- Article shows transformers CAN work on MPS (4.7x speedup!)
- Only unsupported PYTORCH operations cause errors
- We're being overly cautious

**Recommendation:** Use `PYTORCH_ENABLE_MPS_FALLBACK=1` instead to let ops that work, work

---

## Recommended Changes

### Priority 1: Fix Fallback Strategy (Best for M1)
Replace our `SPACY_DISABLE_GPU=1` approach:

```python
# Instead of completely disabling GPU:
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # PyTorch handles unsupported ops

# Then allow MPS to run:
from spacy import prefer_gpu
prefer_gpu()  # MPS will work, unsupported ops fall back to CPU
```

**Benefits:**
- 2-5x speedup for transformer operations that ARE supported
- Automatic fallback for unsupported operations (no crashes)
- Aligns with official Explosion AI recommendation

### Priority 2: Add Thinc to Dependencies
In `pyproject.toml`:
```toml
dependencies = [
    "spacy>=3.7.0,<3.8",
    "thinc>=8.1.0",  # Explicit MPS support
    "numpy<2.0",
    # ...
]
```

### Priority 3: Add spacy-transformers Version Requirement
In transformer-model optional deps:
```toml
[project.optional-dependencies]
transformer-model = [
    "spacy-transformers>=1.1.8",  # MPS support for transformers
    "en-core-web-trf @ https://...",
]
```

### Priority 4: Add Diagnostic Logging (Optional)
```python
# After loading model, verify MPS is active
try:
    from thinc.api import get_current_ops
    ops = get_current_ops()
    logger.info(f"Thinc backend: {ops.__class__.__name__}")
    # Will show "MPSOps" if MPS is active
except ImportError:
    pass  # Older Thinc version
```

### Priority 5: Use Official Error Handling
Replace try/except `prefer_gpu()` with:
```python
try:
    spacy.require_gpu()
except RuntimeError as e:
    logger.warning(f"GPU not available: {e}. Using CPU.")
    # Falls back to CPU automatically
```

---

## Summary: Current Risk Assessment

### ✅ Safe Now
- Won't crash on M1 Macs (our fallback works)
- Dependencies are sufficient versions
- CPU fallback is functional

### ⚠️ Suboptimal
- Disabling MPS entirely means 2-5x performance loss on transformers
- Could allow MPS operations with graceful fallback instead
- Not using article's official `require_gpu()` pattern
- Not explicitly verifying Thinc MPS support

### 🚀 Recommended Direction
Implement `PYTORCH_ENABLE_MPS_FALLBACK=1` approach instead of disabling GPU - gets benefits of MPS while being crash-safe.

---

## Testing Strategy
Once changes are made:
```bash
# Test with transformer model
uv run TextVerbGroupCounter.py input.csv groups.csv output.xlsx \
  --model en_core_web_trf

# Check logs for:
# - "Device: Metal (Apple Silicon)"
# - "Thinc backend: MPSOps" (if diagnostic logging added)
# - No "placeholder storage" errors
```
