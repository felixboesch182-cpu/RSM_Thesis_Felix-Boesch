# Thesis — Code

Code for the master's thesis *"Beyond Patent Counts: Using Natural Language Processing to
Predict Startup Success in Europe"* (Felix Boesch). The pipeline links
European startups (Crunchbase) to their patents (Lens.org), derives NLP-based **novelty** and
**competitive-density** features from patent text using PaECTER embeddings, and tests whether
these add predictive value over traditional OECD patent-quality indicators for funding and exit
outcomes.

> **Data is not included in this repository.** Crunchbase exports are licensed and shared
> separately; all other inputs and derived datasets (CSV, Parquet, embeddings, …) are excluded
> via `.gitignore`. The notebooks document the logic and expect the data files at the paths
> referenced inside each one. The cleaned Crunchbase company/funding tables are provided directly
> as the starting point for stage 02.

All scripts are commented Jupyter notebooks, organised by pipeline stage; notebooks within a
stage are numbered in execution order.

## Pipeline

| Stage | Folder | Purpose |
|---|---|---|
| 02 | `02_matching_crunchbase_data/` | Merge the cleaned Crunchbase companies and funding rounds into the master analytical table. |
| 03 | `03_lens_query_and_ingestion/` | Generate batched Lens.org patent queries and build the patent master table from the Lens exports. |
| 04 | `04_startup_patent_matching/` | Name-match startups to patents / applicants. |
| 05 | `05_NLP_feature_extraction/` | Map Lens IDs to patent families, compute PaECTER embeddings, build the prior-art universe, derive **novelty** + **competitive density**, bridge the OECD quality indicators, and validate the measures. |
| 06 | `06_final_data_set/` | Build the firm-level analytical sample, preprocess, run the models, VIF diagnostics, and generate the result tables and Figure 1. |

### Stage 05 — execution order
Pipeline steps run in numbered order: `01_map_lens_ids_to_families` →
`02_encode_uncovered_patents` → `03_consolidate_embeddings` →
`04_build_prior_art_universe` → `05_build_universe_matrix` → `06_compute_novelty_and_density` →
`07_merge_oecd_quality`. The remaining notebooks are validations and can be run afterwards.

Key notebooks:
- `01_map_lens_ids_to_families` — join Lens IDs to DOCDB patent families.
- `02_encode_uncovered_patents` — encode (with PaECTER) the patents not in the pre-computed corpus.
- `03_consolidate_embeddings` — assemble one embedding per Lens ID (pre-computed + in-house); feeds the embedding-consistency validation.
- `04_build_prior_art_universe` / `05_build_universe_matrix` — build the prior-art universe and its memory-mapped, L2-normalised embedding matrix.
- `06_compute_novelty_and_density` — novelty and competitive density over the five-year prior-art window.
- `07_merge_oecd_quality` — merge the seven OECD patent-quality indicators (family-level mean).
- `10_compute_novelty_excluding_family` — top-1 novelty variant that masks out same-family priors; feeds the competitive-density validation.
- `08_validate_embedding_consistency`, `09_extract_whalen_benchmark`, `11_validate_density_vs_whalen`, `12_validate_novelty_vs_kelly` — construct / embedding-consistency validation of the NLP measures (in-house vs. pre-computed embeddings; Whalen et al. 2020; Kelly et al. 2021). `11_validate_density_vs_whalen` consumes the `10_compute_novelty_excluding_family` output.

### Stage 06 — modelling
- `01_build_startups_with_patents` → `02_build_final_sample_focus` → `03_preprocess_for_ml` build and preprocess the firm-level table (tree-ready + linear-ready versions).
- `04_run_models_and_build_tables` — for two outcomes (cumulative funding, regression; exit,
  classification) and four nested specifications (controls ⊂ +traditional / +NLP ⊂ full), under two
  model families:
  - **gradient-boosted trees (LightGBM)** with nested cross-validation — outer repeated 5-fold ×
    10-seed evaluation, inner 3-fold grid search for the hyperparameters;
  - **penalised linear** (`RidgeCV` / L2-`LogisticRegressionCV`) over the same repeated 5-fold ×
    10-seed splits, with fold-internal winsorising, imputation, and standardisation.

  Paired contrasts use one-sided Nadeau–Bengio-corrected tests. Also emits SHAP importances,
  top-decile screening lift, VIF diagnostics for the traditional block, power / minimum-detectable-effect,
  the result tables, and Figure 1.

## Environment
Python 3.12. Core libraries: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `shap`,
`sentence-transformers` / `torch` (PaECTER embeddings), `pyarrow`, `scipy`. The `7zz` (p7zip)
binary must be on `PATH` to read the OECD archives. The PaECTER model (`mpi-inno-comp/paecter`) is
downloaded from Hugging Face on first use; the embedding and novelty steps run on Apple MPS or CUDA
when available and fall back to CPU.

## Data layout and reproduction

All data lives outside version control under `data/` (git-ignored), split into:

- `data/raw/` — inputs provided externally; treated as read-only.
- `data/processed/` — everything the pipeline generates (intermediate artifacts and results).

Every notebook resolves these two folders independently of where it is run:

```python
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data").exists())
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
```

### Required raw inputs (`data/raw/`)

Organised into subfolders by source:

```
data/raw/
├── crunchbase/    startup_sample_crunchbase_export.csv, funding_rounds_crunchbase_export.csv
├── lens/          *.jsonl[.gz]  (Lens patent export batches)
├── embeddings/    EPO_PaECTER_embeddings.parquet, USPTO_PaECTER_embeddings.parquet,
│                  patent_features_base.parquet
├── bigquery/      family_dates.parquet, bq-results-*.csv  (publication↔family exports)
├── oecd/          202602_OECD_PATENT_QUALITY_EPO_INDIC.7z, …_USPTO_INDIC.7z
└── benchmarks/    most_sim.zip (Whalen et al. 2020), kelly_matched.parquet (Kelly et al. 2021)
```

> **Pre-computed PaECTER embeddings** — the `EPO_PaECTER_embeddings.parquet` and
> `USPTO_PaECTER_embeddings.parquet` files in `data/raw/embeddings/` are not generated by the
> pipeline. Download them from the MPG Edmond repository and place them in that folder:
> <https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.BGRPMI>

> **Licensed data (Crunchbase & Lens.org)** — the `data/raw/crunchbase/` and `data/raw/lens/`
> files are licensed and cannot be redistributed in this repository. Request them from the
> repository owner, or obtain equivalent exports under your own Crunchbase / Lens.org licence.

| Subfolder | Contents | Source | First used by |
|---|---|---|---|
| `crunchbase/` | company + funding-round exports | Crunchbase | `02_matching_crunchbase_data` |
| `lens/` | `*.jsonl[.gz]` patent batches | Lens.org | `03_lens_query_and_ingestion`, `04_startup_patent_matching` |
| `embeddings/` | `EPO_PaECTER_embeddings.parquet`, `USPTO_PaECTER_embeddings.parquet` (MPI), `patent_features_base.parquet` (focal embeddings + priority dates) | PaECTER (MPI) — [download](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.BGRPMI) | `05` (01 / 03 / 04 / 06 / 10) |
| `bigquery/` | `family_dates.parquet`, `bq-results-*.csv` (the OECD bridge and the Kelly bridge with `publication_number` + `family_id`) | BigQuery | `05` (04 / 07 / 12) |
| `oecd/` | OECD Patent Quality Indicator archives (`.7z`) | OECD | `05` (07) |
| `benchmarks/` | `most_sim.zip`, `kelly_matched.parquet` | Whalen et al. 2020; Kelly et al. 2021 | `05` (09 / 12) |

### Run order

Stages run in numeric order `02 → 06`. Within stage 05, notebooks `01`–`07` build the features;
the validations (`08`, `09`, `11`, `12`) run afterwards, with one wrinkle: `11_validate_density_vs_whalen`
**Part A** builds `whalen_match_subsample.parquet`, which `09_extract_whalen_benchmark` needs, so the
effective order is `11`(Part A) → `09` → `11`(Part C). `10_compute_novelty_excluding_family` must run
before `11`.

### Reproduction check

1. Fresh clone; create `data/raw/` and place all required raw inputs above.
2. Empty `data/processed/`.
3. Run the notebooks in order headless, e.g. `jupyter nbconvert --to notebook --execute <nb>`.
4. Confirm every stage regenerates its outputs and that `data/processed/ml_results/` matches the reported tables.
