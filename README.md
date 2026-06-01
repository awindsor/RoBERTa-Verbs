# RoBERTa Verbs

A script-based pipeline for extracting verbs from text, running RoBERTa masked-language-model inference over those verbs in context, mapping MLM predictions to semantic verb groups, and aggregating the results.

Most tools support a command-line mode. Several of the larger tools also launch a PySide6 GUI when run with no arguments.

## Installation

Use the project environment if available:

```bash
uv sync
```

Or install the main dependencies manually:

```bash
pip install spacy transformers torch lemminflect openpyxl PySide6 tqdm
python -m spacy download en_core_web_sm
```

Optional transformer spaCy model support is defined in `pyproject.toml` under the `transformer-model` extra.

## Main Workflow

```text
Raw text or CSV text column
  -> SpaCyVerbExtractor2.py
  -> FilterSpaCyVerbs.py or randomSampleCSV.py
  -> RoBERTaMaskedLanguageModelVerbs.py
  -> MLMGroupAggregator.py
  -> LemmaToGroupProbs.py
```

Typical CLI run:

```bash
python SpaCyVerbExtractor2.py input.csv --csv-text-column text --output verbs.csv
python FilterSpaCyVerbs.py verbs.csv verbs_min20.csv --field lemma --min-freq 20
python RoBERTaMaskedLanguageModelVerbs.py verbs_min20.csv verbs_mlm.csv --model roberta-base --batch-size 8 --top-k 10 --device mps
python MLMGroupAggregator.py verbs_mlm.csv groups.csv verbs_mlm_groups.csv
python LemmaToGroupProbs.py verbs_mlm_groups.csv lemma_group_probs.xlsx --auto-groups
```

Metadata JSON files are written beside outputs where supported. The metadata records settings, source files, checksums, statistics, and reconstructed commands for reproducibility.

## Scripts

### `SpaCyVerbExtractor2.py`

Extracts verbs from one raw text file, one directory of raw text files, or one CSV text column. This is the preferred extractor for the current RoBERTa pipeline because it writes `context` and `span_in_context`.

GUI:

```bash
python SpaCyVerbExtractor2.py
```

CLI:

```bash
python SpaCyVerbExtractor2.py input.txt --output verbs.csv
python SpaCyVerbExtractor2.py texts_folder --output verbs.csv --filter-expr "{{file name}}[:4] == 'week'"
python SpaCyVerbExtractor2.py input.csv --csv-text-column text --csv-row-label-mode row_number --output verbs.csv
python SpaCyVerbExtractor2.py input.csv --csv-text-column text --csv-row-label-mode id_column --csv-id-column row_id --output verbs.csv
python SpaCyVerbExtractor2.py input.csv --csv-text-column text --csv-row-label-mode all_columns --context-mode chars --context-chars 301
python SpaCyVerbExtractor2.py --load-metadata verbs.json
```

Current options:

- Inputs: one positional source, `--paths-file`
- Output: `--output`, `--tsv`
- CSV input: `--csv-text-column`, `--csv-row-label-mode`, `--csv-id-column`
- Filtering: `--filter-expr`
- Context: `--context-mode sentences|chars|all`, `--context-sentences`, `--context-chars`
- NLP/runtime: `--model`, `--encoding`, `--include-aux`, `--log-every`, `--log-level`
- Reproducibility: `--load-metadata`, `--allow-checksum-mismatch`

Filter expressions use `{{column}}` or file placeholders. CSV placeholders are column names. Directory/file placeholders are `{{full path}}`, `{{directory name}}`, `{{file name}}`, and `{{suffix}}`. Supported operations include `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`, `and`, `or`, `not`, parentheses, list/set/tuple/dict literals, and string indexing/slicing.

Important output columns include `lemma`, `surface_lower`, `context`, `span_in_context`, and source-document location fields. Older downstream code may also accept `sentence` and `span_in_sentence_char`.

`SpaCyVerbExtractor2.py` writes preliminary JSON metadata before model loading. When loading metadata from the CLI, input checksums must match by default; pass `--allow-checksum-mismatch` only when intentionally rerunning against changed input files.

### `SpaCyVerbExtractor.py`

Legacy/chunked extractor for very large raw text files. It supports chunk size, overlap, overlap de-duplication, and batch processing with `nlp.pipe()`. Use it when large-file streaming is more important than Extractor2's CSV row labeling and configurable context windows.

```bash
python SpaCyVerbExtractor.py input.txt --output verbs.csv
python SpaCyVerbExtractor.py --paths-file filepaths.txt --output verbs.tsv --tsv
python SpaCyVerbExtractor.py input.csv --csv-text-column text --include-csv-fields --output verbs.csv
```

### `SpaCyVerbCounter.py`

Counts extracted verb values.

```bash
python SpaCyVerbCounter.py verbs.csv lemma_counts.csv --field lemma
python SpaCyVerbCounter.py verbs.csv surface_counts.csv --field surface_lower
```

### `FilterSpaCyVerbs.py`

Filters a verb CSV by frequency, with optional row-level filtering.

```bash
python FilterSpaCyVerbs.py verbs.csv verbs_min20.csv --field lemma --min-freq 20
python FilterSpaCyVerbs.py verbs.csv verbs_mid.csv --field surface_lower --min-freq 10 --max-freq 5000
python FilterSpaCyVerbs.py verbs.csv verbs_top.csv --field lemma --min-freq 80%
python FilterSpaCyVerbs.py verbs.csv verbs_week.csv --field lemma --min-freq 5 --where "{{source_file}}[:4] == 'week'"
python FilterSpaCyVerbs.py --load-metadata verbs_min20.json verbs_min20.csv rerun.csv --strict-checksum
```

Current options:

- Required in CLI mode: `input_csv`, `output_csv`
- Frequency field: `--field lemma|surface_lower`
- Frequency bounds: `--min-freq`, `--max-freq`
- Percentile mode: append `%`, for example `--min-freq 80%`
- Row filter: `--where "{{source}} == 'NCTE'"`
- Reproducibility: `--load-metadata`, `--strict-checksum`

Row filters support `{{column}}` placeholders with safe Python-style boolean expressions: comparisons, membership tests, `and`/`or`/`not`, collection literals, and string indexing/slicing such as `{{lemma}}[:3] == 'run'`.

### `randomSampleCSV.py`

Samples rows from a CSV.

```bash
python randomSampleCSV.py verbs.csv verbs_sample.csv 100000 --seed 42
```

### `RoBERTaMaskedLanguageModelVerbs.py`

Runs masked-language-model inference over each verb row.

GUI:

```bash
python RoBERTaMaskedLanguageModelVerbs.py
```

CLI:

```bash
python RoBERTaMaskedLanguageModelVerbs.py verbs_min20.csv verbs_mlm.csv --model roberta-base --batch-size 8 --top-k 10
python RoBERTaMaskedLanguageModelVerbs.py verbs_min20.csv verbs_mlm.csv --device mps --log-every 5000
python RoBERTaMaskedLanguageModelVerbs.py verbs_min20.csv verbs_mlm_with_context.csv --include-context
python RoBERTaMaskedLanguageModelVerbs.py --load-metadata verbs_min20.json verbs_min20.csv verbs_mlm.csv
python RoBERTaMaskedLanguageModelVerbs.py verbs_min20.csv debug_mlm.csv --debug-limit 100 --log-level DEBUG
```

Current options:

- Required in CLI mode: `input_csv`, `output_csv`
- Model/runtime: `--model`, `--batch-size`, `--top-k`, `--device`
- Logging/debugging: `--log-every`, `--log-level`, `--debug-limit`
- CSV/reproducibility: `--encoding`, `--include-context`, `--load-metadata`

Input CSVs must contain either `context` + `span_in_context` or `sentence` + `span_in_sentence_char`. By default, output omits the text column used for masking (`context` or `sentence`) and appends `token_1`, `prob_1`, ..., `token_k`, `prob_k`. Use `--include-context` or the GUI checkbox to retain the context/text column in the MLM output.

Memory note: if macOS reports application memory pressure, lower `--batch-size`. The script retries smaller batches on memory failures and clears CUDA/MPS caches between batches, but RoBERTa still produces large logits internally.

### `MLMGroupAggregator.py`

Maps MLM token predictions to semantic group probabilities without rerunning RoBERTa.

GUI:

```bash
python MLMGroupAggregator.py
```

CLI:

```bash
python MLMGroupAggregator.py verbs_mlm.csv groups.csv verbs_mlm_groups.csv
python MLMGroupAggregator.py verbs_mlm.csv groups.csv groups_short.csv --short --include-count
python MLMGroupAggregator.py verbs_mlm.csv groups.csv verbs_mlm_groups.csv --workers 4 --chunk-size 2000
python MLMGroupAggregator.py verbs_mlm.csv groups.csv verbs_mlm_groups.csv --load-metadata verbs_mlm.json
```

Current options:

- Required in CLI mode: `mlm_csv`, `group_csv`, `output_csv`
- Prediction parsing: `--top-k`, `--lemma-col`
- Output shape: `--short`, `--include-count`
- Runtime/logging: `--workers`, `--chunk-size`, `--log-every`, `--log-level`
- CSV/reproducibility: `--encoding`, `--load-metadata`

Group CSV format uses group names as headers and lemmas underneath each group:

```csv
thinking,motion,communication
think,run,say
believe,jump,tell
```

### `LemmaToGroupProbs.py`

Aggregates row-level group probabilities by lemma. Outputs `.csv`, `.tsv`, `.xlsx`, or `.xlsm`.

```bash
python LemmaToGroupProbs.py verbs_mlm_groups.csv lemma_group_probs.csv
python LemmaToGroupProbs.py verbs_mlm_groups.csv lemma_group_probs.xlsx
python LemmaToGroupProbs.py verbs_mlm_groups.csv lemma_group_probs.xlsx --second-threshold 0.40
python LemmaToGroupProbs.py verbs_mlm_groups.csv lemma_group_probs.xlsx --auto-groups --importance-cutoff 0.05
python LemmaToGroupProbs.py verbs_mlm_groups.csv lemma_group_probs.csv --group-cols thinking motion communication
```

Current options:

- Required in CLI mode: `input_csv`, `output`
- Column selection: `--lemma-col`, `--group-cols`
- Excel highlighting/analysis: `--second-threshold`, `--auto-groups`, `--importance-cutoff`, `--include-ambiguous-auto-groups`, `--overlap-measure`, `--vba-links`
- CSV/reproducibility: `--encoding`, `--load-metadata`

### `MLMLemmaExtractor.py`

Prints sentences for one or more lemmas from an MLM output CSV.

```bash
python MLMLemmaExtractor.py think verbs_mlm.csv
python MLMLemmaExtractor.py think,believe verbs_mlm.csv --limit 25
python MLMLemmaExtractor.py run verbs_mlm.csv --show-topk --top-k 5
```

Current options:

- Required: `lemmas`, `mlm_csv`
- Columns: `--lemma-col`, `--sentence-col`
- Display: `--limit`, `--show-topk`, `--top-k`
- CSV: `--encoding`

### `tagged_verbs_to_groups.py`

Converts a manually tagged lemma table into the group CSV format used by `MLMGroupAggregator.py`.

```bash
python tagged_verbs_to_groups.py tagged_verbs.csv groups.csv
python tagged_verbs_to_groups.py tagged_verbs.csv groups.csv --ignore uncertain misc
```

Input format:

```csv
lemma,frequency,thinking,motion
think,100,1,
run,50,,1
```

### `TextVerbGroupCounter.py`

Counts verb groups directly from a CSV text column.

```bash
python TextVerbGroupCounter.py documents.csv groups.csv document_group_counts.xlsx --text-col text
python TextVerbGroupCounter.py documents.csv groups.csv document_group_counts.csv --model en_core_web_sm --include-aux
python TextVerbGroupCounter.py documents.csv groups.csv counts.xlsx --where "{{grade}} == '8'"
```

Current options:

- Required: `input_csv`, `group_csv`, `output`
- Input/NLP: `--text-col`, `--encoding`, `--model`, `--include-aux`, `--batch-size`
- Filtering/runtime: `--where`, `--force-cpu`
- Reproducibility: `--load-metadata`

### `SpellChecker.py`

Spell-checks text or CSV input and writes corrected output plus metadata.

```bash
python SpellChecker.py input.txt corrected.txt
python SpellChecker.py rows.csv corrected.csv --text-column text --csv-format complete
python SpellChecker.py input.txt corrected.patch --text-format patch
```

Current options:

- Required in CLI mode: `input_file`, `output_file`
- Mode/dictionaries: `--mode`, `--language`, `--custom-dict`, `--ignore-list`
- CSV/text handling: `--text-column`, `--csv-format`, `--text-format`, `--encoding`
- Reproducibility: `--load-metadata`, `--strict-checksum`

## Utility Scripts

- `UnifiedVerbToolsApp.py`: GUI launcher for multiple tools.
- `run_with_mps.py`: helper for running with Apple Metal/MPS settings.
- `merge_csv_by_row_number.py`: joins CSVs by row number.
- `FilteredVerbInflectionExtractor.py`: extracts inflection lists from filtered verb data.
- `count_patterns.py`: local pattern-counting utility for targeted analysis.
- `view_log.py`: log viewer utility.

Experimental or legacy files are present in the repository, including `MLMGroupProbabilityAggregator.py` and several one-off scripts with descriptive filenames. Prefer the scripts listed above for current pipeline work.

## Metadata And Checksums

Supported scripts write a JSON sidecar with the same base name as the output file, for example `verbs_mlm.csv` -> `verbs_mlm.json`.

Checksum behavior:

- Extraction metadata stores input file checksums and output checksum.
- Filter metadata stores both input and output checksums.
- MLM metadata verifies upstream filter/extractor output checksums when chaining from metadata.
- Aggregator metadata stores checksums for the MLM CSV, group CSV, and output.

Use `--load-metadata` to reload settings from a previous run. CLI arguments supplied at the same time override loaded settings where supported.

For `SpaCyVerbExtractor2.py`, CLI metadata loading aborts on missing, changed, or unlisted input files unless `--allow-checksum-mismatch` is supplied.

## Testing

Run syntax checks for edited scripts:

```bash
python3 -m py_compile SpaCyVerbExtractor2.py FilterSpaCyVerbs.py RoBERTaMaskedLanguageModelVerbs.py MLMGroupAggregator.py LemmaToGroupProbs.py
```

Run the test suite:

```bash
uv run pytest -q
```

Some tests or direct script runs require project dependencies such as `spacy`, `torch`, `lemminflect`, `openpyxl`, and `PySide6`.
