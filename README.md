# G9a Inhibitor ML Pipeline

Structured, script-based reproduction of
[Application of machine learning for predicting G9a inhibitors](https://doi.org/10.1039/d4dd00101j)
(Ivanova et al., *Digital Discovery*, 2024).

Original notebooks live in `G9a_clsf/` (unchanged). This project turns that work into
runnable pipeline steps that write metrics/models to `outputs/`.

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

## Pipeline steps

| Step | Script | Output |
|------|--------|--------|
| 00 | `scripts/00_check_data.py` | Status of raw files |
| 01 | `scripts/01_download_coordinates.py` | Missing target_0 coordinate JSONs via PubChem |
| **create** | `scripts/create_dataset.py` | Clean CSV from downloads/cache (+ optional SMOTE/RUS) |
| **eda** | `scripts/eda_dataset.py` | EDA tables + figures for the CSV |
| 04 | `scripts/04_compare_classifiers.py` | Holdout + CV metrics CSVs |
| 05 | `scripts/05_feature_importance.py` | Importance tables + incremental accuracy plot |
| 06 | `scripts/06_train_final.py` | Final RF metrics JSON + `outputs/models/final_rf.joblib` |

### Create dataset + EDA (recommended after download)

```bash
python scripts/create_dataset.py              # SMOTE by default
python scripts/create_dataset.py --balancing rus
python scripts/create_dataset.py --balancing none

python scripts/eda_dataset.py
python scripts/eda_dataset.py --csv data/interim/with_solubility_unbalanced.csv
```

Outputs:
- `data/interim/with_solubility_unbalanced.csv`
- `data/processed/with_solubility_smote.csv`
- `outputs/eda/` (tables + markdown/JSON summary)
- `outputs/figures/eda/` (plots)

### Run legacy numbered steps

```bash
python scripts/04_compare_classifiers.py
```

### Run everything

```bash
# Full data (PubChem download of ~39k target_0 compounds can take a while)
python scripts/run_all.py

# Resume from step 3
python scripts/run_all.py --from-step 3

# Skip download if coordinate JSONs already exist
python scripts/run_all.py --skip-download

# Smoke-test download of first 200 target_0 IDs only
python scripts/01_download_coordinates.py --limit 200
```

### Switch balancing (Dataset 3 = RUS)

```bash
python scripts/03_balance.py --balancing rus
# then edit configs/default.yaml balancing: rus  OR pass the same after changing config
```

Or set in `configs/default.yaml`:

```yaml
dataset:
  solubility: with
  balancing: rus
```

## Metrics location

After step 04–06:

```
outputs/metrics/with_solubility_smote_holdout_metrics.csv
outputs/metrics/with_solubility_smote_cv_metrics.csv
outputs/metrics/with_solubility_smote_final_model.json
outputs/figures/with_solubility_smote_incremental_accuracy.png
outputs/models/final_rf.joblib
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

If those two are missing, step 01 downloads them from PubChem using the SID/CID
lists in `data_for_one_column_target_0.csv` (~39k compounds). Downloads are
cached under `data/raw/with_solubility/.cache_*` and can be resumed.

After a **full** download finishes, rebuild and train:

```bash
python scripts/run_all.py --skip-download
# or step-by-step:
python scripts/02_build_features.py
python scripts/03_balance.py
python scripts/04_compare_classifiers.py
python scripts/05_feature_importance.py
python scripts/06_train_final.py --sweep-depth
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
  scripts/                  # CLI steps
  data/raw/with_solubility/ # PubChem inputs (G9a_clsf excluded from git)
  data/interim|processed/
  outputs/metrics|models|figures|eda/
  G9a_clsf/                 # optional local paper clone (gitignored)
```

## Citation

Mariya L. Ivanova et al., *Digital Discovery*, 2024, 3, 2010–2018.
https://doi.org/10.1039/d4dd00101j
