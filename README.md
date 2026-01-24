# RoBERTa Verbs Analysis Workflow

A comprehensive pipeline for extracting verbs from text, computing masked language model predictions using RoBERTa, and organizing results by semantic verb groups.

## Overview

This workflow enables analyzing verb usage patterns and semantic relationships by:
1. Extracting verbs from raw text using spaCy
2. Running RoBERTa masked-language-model (MLM) inference to get contextual predictions
3. Organizing predictions into semantic verb groups
4. Computing aggregated statistics by lemma and group

## Core Components

### Phase 1: Verb Extraction

#### [SpaCyVerbExtractor.py](SpaCyVerbExtractor.py)
Extracts verbs from raw text documents using spaCy NLP pipeline. Supports both CLI and GUI modes with run reconstruction via metadata.

**Features:**
- Chunked processing with overlapping character windows (prevents sentence boundary loss)
- Handles multiple input files via CLI or paths file
- Deduplicates sentences across chunk overlaps
- Outputs lemma, surface form, character span, and full sentence
- **NEW:** Generates JSON metadata with MD5 checksums for complete run reconstruction
- **NEW:** Supports loading previous run settings from JSON metadata

**Output Files:**
- CSV/TSV file with extracted verbs
- `.json` metadata file containing:
  - All extraction settings (model, chunk size, encoding, etc.)
  - MD5 checksums of input and output files
  - Extraction statistics (documents, chunks, sentences, verbs)
  - Timestamp of extraction

**Output Columns:**
- `doc_path`, `chunk_start_char`, `sent_start_char_in_doc`, `sent_index_in_doc_approx`
- `token_index_in_sent`, `lemma`, `surface_lower`, `span_in_sentence_char`, `sentence`

**Usage (GUI):**
```bash
python SpaCyVerbExtractor.py
# In GUI: Load settings from previous run, verify input checksums, configure, and run
```

**Usage (CLI):**
```bash
# Basic extraction
python SpaCyVerbExtractor.py input.txt -o verbs.csv

# With options
python SpaCyVerbExtractor.py --paths-file paths.txt -o verbs.tsv --tsv --include-aux

# Load settings from previous run (CLI options override)
python SpaCyVerbExtractor.py --load-metadata verbs.json input.txt -o verbs2.csv

# Reconstruct exact run
python SpaCyVerbExtractor.py --load-metadata verbs.json
```

**Metadata Features:**
- **Verification:** CLI and GUI verify input file checksums against saved metadata
- **Override:** CLI command-line arguments override loaded settings
- **GUI Changes:** GUI allows changing any settings after loading metadata
- **Filename:** Metadata JSON has same base name as output (e.g., `verbs.csv` → `verbs.json`)

**Requirements:**
```bash
pip install spacy
python -m spacy download en_core_web_sm
pip install PySide6  # GUI mode only
```

---

#### [SpaCyVerbCounter.py](SpaCyVerbCounter.py)
Aggregates verb frequencies from extraction output.

**Output:** CSV with fields and their frequency counts (sorted by frequency descending)

**Usage:**
```bash
python SpaCyVerbCounter.py input.csv output.csv --field lemma
python SpaCyVerbCounter.py input.csv output.csv --field surface_lower
```

---

### Phase 2: Filtering & Sampling

#### [filterSpaCyVerbs.py](filterSpaCyVerbs.py)
Filters verbs by frequency in a two-pass streaming process.

**Features:**
- Memory-efficient: counts unique values but streams rows
- Filters on `lemma` or `surface_lower` with min/max frequency bounds
- Useful for removing rare or overly common verbs

**Usage:**
```bash
python filterSpaCyVerbs.py input.csv output.csv --field lemma --min-freq 10 --max-freq 5000
```

---

#### [randomSampleCSV.py](randomSampleCSV.py)
Randomly samples n rows from a large CSV file without loading entire file into memory.

**Features:**
- Two-pass approach: count rows, then select n distinct indices
- Optional seed for reproducibility

**Usage:**
```bash
python randomSampleCSV.py input.csv output.csv 100000 --seed 42
```

---

### Phase 3: RoBERTa MLM Inference

#### [roberta_mlm_on_verbs.py](roberta_mlm_on_verbs.py)
Runs RoBERTa masked-language-model inference on each verb in context.

**Core Workflow:**
1. For each row: replace verb span with `<mask>` token
2. Run MLM inference via RoBERTa
3. Collect top-k predictions with probabilities
4. (Optional) Sum probabilities for each semantic group

**Output Columns:**
- Original columns from input CSV
- `token_1`, `prob_1`, `token_2`, `prob_2`, ..., `token_k`, `prob_k`
- (If using groups) `group_1`, `group_2`, ... (aggregated probabilities)

**Usage:**
```bash
python roberta_mlm_on_verbs.py verbs.csv verbs_with_mlm.csv --model roberta-base --batch-size 16 --top-k 10
python roberta_mlm_on_verbs.py verbs.csv verbs_with_mlm.csv --group-csv verb_groups.csv --batch-size 32
python roberta_mlm_on_verbs.py verbs.csv out.csv --log-level DEBUG --debug-limit 100
```

**Requirements:**
```bash
pip install transformers torch lemminflect
```

**Features:**
- Streaming-friendly: reads & writes line-by-line
- Supports batch inference for efficiency
- Lemmatizes predictions using lemminflect (VERB-preferred)
- Groups predictions by semantic category if group CSV provided

---

### Phase 4: Group Management & Remapping

#### [mlm_to_groups.py](mlm_to_groups.py)
Recomputes group probabilities using a NEW grouping definition without re-running the model.

**Use Case:**
- You have MLM output with top-k predictions
- You now have a new/different group CSV
- Recompute group-aggregated probabilities from existing predictions

**Group CSV Format:**
```
group1,group2,group3
lemma1,lemma2,
lemma3,,lemma4
```
(Columns = group names; cells = lemmas in that group)

**Output Modes:**
1. **Default:** All original columns + new group columns
2. **`--short`:** Only lemma + group columns

**Usage:**
```bash
python mlm_to_groups.py mlm_out.csv new_groups.csv mlm_out_regrouped.csv
python mlm_to_groups.py mlm_out.csv new_groups.csv regrouped_short.csv --short --include-count
```

**Requirements:**
```bash
pip install lemminflect
```

---

#### [tagged_verbs_to_groups.py](tagged_verbs_to_groups.py)
Transposes manual verb-to-group annotations into a group-by-lemma matrix.

**Input CSV Format:**
```
lemma,frequency,group1,group2,group3
think,100,1,,
run,50,,1,
jump,30,,,1
```
(At most one '1' per row; blanks/0s elsewhere)

**Output CSV Format:**
```
group1,group2,group3
think,,
,run,
,,jump
```
(One row per lemma, ragged columns padded with blanks)

**Usage:**
```bash
python tagged_verbs_to_groups.py tagged_input.csv group_output.csv
python tagged_verbs_to_groups.py tagged_input.csv group_output.csv --ignore think,run
```

---

### Phase 5: Statistical Aggregation

#### [lem_to_group_prob.py](lem_to_group_prob.py)
Aggregates group probabilities by lemma, producing summary statistics.

**Input:** MLM output CSV with group probability columns

**Output Formats:**

**CSV Mode:**
```
lemma,group1(%),group2(%),group3(%),count
think,45.2,32.1,22.7,125
run,18.9,61.3,19.8,98
```

**Excel Mode** (`.xlsx`):
1. **Sheet 1 (lemma_to_groups):**
   - Mean group probabilities (formatted as %)
   - Highest percentage bolded per row
   - Lemma cell **BLUE** if runner-up group ≥ 50% of leader

2. **Sheet 2 (groups_ranked):**
   - For each group: lemmas ranked by descending probability
   - Lemma bolded where it achieves group maximum

**Features:**
- Auto-detects group columns (all after last `prob_k` column)
- Customizable group column specification
- Runner-up detection for ambiguous lemmas

**Usage:**
```bash
python lem_to_group_prob.py mlm_out.csv output.csv
python lem_to_group_prob.py mlm_out.csv output.xlsx  # Excel output
python lem_to_group_prob.py mlm_out.csv output.xlsx --second-threshold 0.40
python lem_to_group_prob.py mlm_out.csv output.csv --group-cols group1,group2,group3
```

**Requirements for Excel:**
```bash
pip install openpyxl
```

---

#### [mlm_lem.py](mlm_lem.py)
Retrieves and displays sentences for a specific lemma from MLM output.

**Features:**
- Filters rows by target lemma
- Optionally displays top-k predictions stored in MLM output
- Configurable output limit

**Usage:**
```bash
python mlm_lem.py run mlm_out.csv
python mlm_lem.py run mlm_out.csv --limit 10
python mlm_lem.py run mlm_out.csv --show-topk --top-k 5
```

---

### Utility: Pattern Analysis

#### [count_patterns.py](count_patterns.py)
Analyzes local context patterns for a specific verb lemma.

**Workflow:**
1. Load verb inflections for target lemma (e.g., "think")
2. Search MLM output for sentences containing any inflection
3. Extract local context windows (n-grams around verb)
4. Count and rank patterns by frequency

**Usage:**
```python
# Edit script to set target verb, then run:
python count_patterns.py
```

---

## Data Files

### Input Data
- `verbs.csv` – Raw extracted verbs (lemma, surface_lower, sentence, span)
- `verb_groups.csv` – Manual verb-to-group mapping (columns = groups)
- `filepaths.txt` – Newline-separated paths to raw text files (for SpaCyVerbExtractor)

### Intermediate/Output Data
- `verbs_sample.csv` – Random sample of extracted verbs
- `verbs_min10.csv` – Verbs filtered to frequency ≥ 10
- `verbs_with_mlm.csv` – MLM predictions + probabilities
- `verbs_with_mlm_2.csv` – Re-grouped version or variant
- `lem_gp_prob.csv` – Aggregated lemma → group probabilities (CSV)
- `lem_gp_prob.xlsx` – Aggregated lemma → group probabilities (Excel, multi-sheet)
- `groups.csv`, `groups_2.csv` – Different semantic verb groupings
- `verb_counts.csv` – Frequency counts for all verbs
- `lemma_counts.csv` – Frequency counts by lemma
- `moved_think.csv` – Analysis output for specific verb

---

## Typical Workflow

```
Raw Text Files (filepaths.txt)
         ↓
    [SpaCyVerbExtractor] → verbs.csv
         ↓
   [filterSpaCyVerbs or randomSampleCSV]
         ↓
    verbs_filtered.csv
         ↓
  [roberta_mlm_on_verbs]
         ↓
    verbs_with_mlm.csv  (token predictions + optional group probs)
         ↓
   [lem_to_group_prob]
         ↓
    lem_gp_prob.csv / .xlsx  (aggregated statistics)
         
Alternative/Parallel:
    [tagged_verbs_to_groups] → Convert manual tags to group matrix
         ↓
    [mlm_to_groups] → Recompute probabilities with new grouping
```

---

## Dependencies

**Core:**
```bash
pip install spacy transformers torch lemminflect
```

**Model:**
```bash
python -m spacy download en_core_web_sm
# RoBERTa model downloaded automatically via transformers
```

**Optional (Excel support):**
```bash
pip install openpyxl
```

---

## Notes

- All scripts stream CSV data when possible to minimize memory usage
- Lemmatization uses `lemminflect` with VERB part-of-speech preference
- Group files are expected to be ragged CSVs (variable column lengths per row)
- Character spans are zero-indexed, end-exclusive (`start:end` format)
- RoBERTa predictions are normalized and lemmatized for group matching
