# G9a Inhibitor ML Pipeline

Structured, script-based reproduction of
[Application of machine learning for predicting G9a inhibitors](https://doi.org/10.1039/d4dd00101j)
(Ivanova et al., *Digital Discovery*, 2024).

Original notebooks live in `G9a_clsf/` (unchanged). This project turns that work into
numbered pipeline steps (`00`–`07`) that write metrics/models to `outputs/`.

## Default experiment (paper Dataset 4)

| Setting | Value |
|---------|--------|
| Solubility | with (AID 1996 @ pH 7.4) |
| Balancing | SMOTE |
| Best model | RandomForest, 5 features, `max_depth=20` |
| Paper accuracy | ~90% (non-overfit) |

Final features: `Volume_1`, `S_relative`, `S`, `N_relative`, `C_relative`.

## Setup

```bash
cd D:\Sai\3_Material\Semester_7\MLE\Project\impl
python -m venv .venv
.venv\Scripts\activate
pip install -e .
# optional ANN extras from the paper:
pip install -e ".[ann]"
```

## End-to-end pipeline (steps 00 → 07)

Run in order. Each step depends on the previous one’s outputs.

| Step | Script | What it does | Main output |
|------|--------|--------------|-------------|
| **00** | `scripts/00_check_data.py` | Check which raw files are present / missing | Console status (exit 1 if coords missing) |
| **01** | `scripts/01_download_coordinates.py` | Download missing target_0 2D/3D coords from PubChem | `SID_2D_target_0.json`, `CID_3D_target_0.json` (+ `.cache_*`) |
| **02** | `scripts/02_build_features.py` | Engineer features from PubChem tables + coords | `data/interim/with_solubility_unbalanced.csv` |
| **03** | `scripts/03_balance.py` | Balance classes (SMOTE or RUS) | `data/processed/with_solubility_smote.csv` |
| **04** | `scripts/04_eda_dataset.py` | Exploratory data analysis on the CSV | `outputs/eda/`, `outputs/figures/eda/` |
| **05** | `scripts/05_compare_classifiers.py` | Train/compare 5 classifiers (holdout + CV) | `outputs/metrics/*_holdout_metrics.csv`, `*_cv_metrics.csv` |
| **06** | `scripts/06_feature_importance.py` | RF / permutation / chi² importance + plot | Importance CSVs + incremental accuracy figure |
| **07** | `scripts/07_train_final.py` | Train paper’s final 5-feature RandomForest | `outputs/models/final_rf.joblib` + metrics JSON |

**Optional shortcut (replaces steps 02 + 03):**

```bash
python scripts/create_dataset.py              # SMOTE by default (from config)
python scripts/create_dataset.py --balancing rus
python scripts/create_dataset.py --balancing none
```

**Orchestrator:** `scripts/run_all.py` runs steps 00–07 in sequence.

### Flow diagram

```text
00 check raw files
        │
        ▼
01 download PubChem coordinates (if missing)
        │
        ▼
02 build engineered features  ──┐
        │                       │  (or: create_dataset.py)
        ▼                       │
03 balance (SMOTE / RUS)     ◄──┘
        │
        ▼
04 EDA (tables + figures)
        │
        ▼
05 compare classifiers
        │
        ▼
06 feature importance
        │
        ▼
07 train final RF model
```

## How to run

### Option A — one command (full pipeline)

```bash
# Full data (PubChem download of ~39k target_0 compounds can take a long time)
python scripts/run_all.py

# Skip download if coordinate JSONs already exist
python scripts/run_all.py --skip-download

# Resume from a given step (e.g. after download finished)
python scripts/run_all.py --from-step 2

# Train with depth sweep on the final model
python scripts/run_all.py --skip-download --sweep-depth

# Smoke-test: download only first 200 IDs, then continue
python scripts/run_all.py --download-limit 200
```

### Option B — step by step

```bash
python scripts/00_check_data.py
python scripts/01_download_coordinates.py          # omit if coords already OK
# python scripts/01_download_coordinates.py --limit 200   # smoke-test only

python scripts/02_build_features.py
python scripts/03_balance.py                       # or: --balancing rus
# OR: python scripts/create_dataset.py --balancing smote

python scripts/04_eda_dataset.py
python scripts/05_compare_classifiers.py
python scripts/06_feature_importance.py
python scripts/07_train_final.py --sweep-depth
```

### Switch balancing (Dataset 3 = RUS)

```bash
python scripts/03_balance.py --balancing rus
```

Or set in `configs/default.yaml`:

```yaml
dataset:
  solubility: with
  balancing: rus
```

Then re-run from step 03 onward.

## Metrics & model location

After steps 05–07:

```
outputs/metrics/with_solubility_smote_holdout_metrics.csv
outputs/metrics/with_solubility_smote_cv_metrics.csv
outputs/metrics/with_solubility_smote_final_model.json
outputs/figures/with_solubility_smote_incremental_accuracy.png
outputs/models/final_rf.joblib
```

EDA (step 04):

```
outputs/eda/
outputs/figures/eda/
```

## Raw data notes

Required raw inputs live in `data/raw/with_solubility/` (copied out of the
original `G9a_clsf/` paper clone, which is **gitignored** and not part of this
repo).

Included (tracked or local):

- labeled targets: `data_for_one_column_target_{0,1}.csv`
- PubChem properties: `pubChemComputed_target_{0,1}.csv`
- solubility table: `pubchem_solubility.csv`
- coordinate JSONs for target_1 (small enough to keep locally)

**Large files (gitignored — present locally after copy/download):**

- `SID_2D_target_0.json` (~57 MB)
- `CID_3D_target_0.json` (~250 MB)

If those two are missing, step **01** downloads them from PubChem using the SID/CID
lists in `data_for_one_column_target_0.csv` (~39k compounds). Downloads are
cached under `data/raw/with_solubility/.cache_*` and can be resumed.

After a **full** download finishes:

```bash
python scripts/run_all.py --skip-download --from-step 2 --sweep-depth
```

Smoke-test only (first N target_0 IDs — **not** paper-scale metrics):

```bash
python scripts/01_download_coordinates.py --force --limit 200
python scripts/run_all.py --skip-download --from-step 2
```

Then re-run without `--limit` (use `--force`) before claiming paper results.

The without-solubility notebooks need extra files. The default pipeline focuses
on Dataset 4 (best paper result).

## Project layout

```
impl/
  configs/default.yaml
  src/g9a_ml/               # library
  scripts/                  # numbered CLI steps 00–07
  data/raw/with_solubility/ # PubChem inputs (G9a_clsf excluded from git)
  data/interim|processed/
  outputs/metrics|models|figures|eda/
  G9a_clsf/                 # optional local paper clone (gitignored)
```

## Citation

Mariya L. Ivanova et al., *Digital Discovery*, 2024, 3, 2010–2018.
https://doi.org/10.1039/d4dd00101j
