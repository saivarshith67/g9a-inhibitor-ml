# Predicting G9a Inhibitors with Machine Learning

**Project presentation brief**  
Semester 7 · MLE Project · Implementation & restructuring of Ivanova et al. (2024)

---

## Slide 1 — Title

**Application of Machine Learning for Predicting G9a Inhibitors**

- Based on: Ivanova, M. L., Russo, N., Djaid, N., & Nikolic, K. (2024). *Digital Discovery*, 3, 2010–2018  
- DOI: [https://doi.org/10.1039/d4dd00101j](https://doi.org/10.1039/d4dd00101j)  
- Goal of this project: turn the paper’s Jupyter notebooks into a **reproducible, step-based ML pipeline** and produce datasets, EDA, and metrics without running notebooks one by one

---

## Slide 2 — Motivation & problem

**Why G9a?**

- G9a (EHMT2) is an epigenetic regulator linked to gene expression and cancer biology  
- Lab screening of compounds is slow and expensive  
- PubChem already holds large bioassay + chemical property data  

**ML task**

- **Binary classification**: is a compound an **active G9a inhibitor** (target = 1) or not (target = 0)?  
- Labels from PubChem AID **504332** (qHTS assay)  
- Optional water solubility at pH 7.4 from AID **1996**

**Paper highlight**

- Best model: **Random Forest**, **5 features**, ~**90%** accuracy (non-overfit) on Dataset 4 (with solubility + SMOTE)

---

## Slide 3 — What we inherited vs what we built

| Original (`G9a_clsf/`) | Our project (`impl/`) |
|------------------------|------------------------|
| Multiple Jupyter notebooks | Numbered CLI scripts + library package |
| Manual step-by-step cells | One command per pipeline stage |
| Metrics printed in notebooks | Metrics/figures saved under `outputs/` |
| Incomplete coordinate files in repo | Full PubChem download + cache resume |
| Hard to re-run | Configurable YAML + repeatable CSVs |

**Design principle:** keep original code/data intact; build a clean ML project beside it.

---

## Slide 4 — Project objectives (work done)

1. Restructure paper code into a proper ML project layout  
2. Download missing PubChem **2D/3D coordinate** data for the majority class  
3. Build a **clean CSV dataset** from raw tables + engineered features  
4. Run **EDA** with tables and plots suitable for reporting  
5. Provide scripts to **train, compare classifiers, feature importance, and final model**  
6. Document everything so the pipeline can be demoed and presented

---

## Slide 5 — Paper methodology (high level)

```text
PubChem AID 504332 ──► labels (active inhibitor vs other)
        │
        ├── PubChem computed properties (MW, TPSA, XLogP, …)
        ├── 2D substance + 3D compound coordinates
        ├── Engineered features (volumes, atom ratios, SMILES similarity)
        └── Optional solubility (AID 1996)
                │
                ▼
     Balance (RUS or SMOTE) ──► 5 dataset variants
                │
                ▼
   Train DTC / RF / GBC / XGB / SVM (+ ANN, PySpark in paper)
                │
                ▼
   Feature importance ──► reduce to 5 features ──► final RF (~90%)
```

---

## Slide 6 — Five datasets from the paper

| ID | Solubility | Balancing | Approx. size (paper) |
|----|------------|-----------|----------------------|
| D1 | No | RUS | ~54.8k × 60 |
| D2 | No | SMOTE | ~80k × 60 |
| D3 | Yes | RUS | ~7.8k × 61 |
| **D4** | **Yes** | **SMOTE** | **~75.3k × 61** ← best |
| D5 | No | SMOTE (big) | ~613k × 60 |

**Our default focus:** Dataset **D4** (with solubility + SMOTE) — paper’s best-performing setup.

---

## Slide 7 — Project structure

```text
impl/
├── G9a_clsf/                 # Original paper notebooks + raw data
├── configs/default.yaml      # Dataset, train, final-model settings
├── src/g9a_ml/               # Reusable library
│   ├── data/                 # load, download, assemble cache, build, balance
│   ├── features/             # coordinates, volumes, composition, similarity
│   └── models/               # classifiers, train, feature importance
├── scripts/                  # CLI entry points
├── data/
│   ├── interim/              # unbalanced engineered CSV
│   └── processed/            # balanced CSV + metadata
└── outputs/
    ├── eda/                  # EDA tables + summary
    ├── figures/              # plots
    ├── metrics/              # model metrics
    └── models/               # saved estimators
```

---

## Slide 8 — End-to-end pipeline (scripts)

| Stage | Script | Purpose |
|-------|--------|---------|
| Check | `00_check_data.py` | Verify raw files present / missing |
| Download | `01_download_coordinates.py` | Fetch missing target_0 2D/3D coords (resumable) |
| **Dataset** | `create_dataset.py` | Build proper CSV from downloads + features |
| **EDA** | `eda_dataset.py` | Explore the generated CSV |
| Compare | `04_compare_classifiers.py` | Holdout + CV metrics for 5 models |
| Importance | `05_feature_importance.py` | RF / permutation / chi² + incremental accuracy |
| Final | `06_train_final.py` | 5-feature RF (paper final model) |
| All | `run_all.py` | Orchestrate steps |

---

## Slide 9 — Data gap we solved

Upstream GitHub repo was **missing**:

- `SID_2D_target_0.json`  
- `CID_3D_target_0.json`  

These are required for feature engineering on the majority class (~39k compounds with solubility).

**What we did**

- Built a **batch + parallel PubChem downloader**  
- Cached batches under `.cache_SID_2D_target_0/` and `.cache_CID_3D_target_0/`  
- Assembled final JSONs  

**Downloaded (full)**

| File | Size |
|------|------|
| `SID_2D_target_0.json` | ~59 MB |
| `CID_3D_target_0.json` | ~262 MB |
| Compounds requested | **38,885** |

---

## Slide 10 — Feature engineering (what goes into the CSV)

**From PubChem**

- Molecular weight, TPSA, XLogP3, heavy-atom / H-bond counts, rotatable bonds, etc.  
- Water solubility at pH 7.4 (Dataset 4)

**Engineered (paper formulas)**

- 2D / 3D coordinate spans and skewness  
- Hypothetical volumes (`Volume_1`, `Volume_2`, 3D volumes)  
- Relative atom proportions (`C_relative`, `S_relative`, …)  
- Mass proportions (`C`, `S`, `N`, …)  
- Size ratios along axes  
- SMILES **Tanimoto similarity to lysine** (RDKit)

**Preprocessing**

- Keep uncharged compounds only  
- Drop IDs / SMILES / formula after feature generation  
- Shift selected signed columns (+20) as in the original notebooks  

---

## Slide 11 — Dataset we created

**Command**

```bash
python scripts/create_dataset.py --balancing smote
```

**Results**

| Artifact | Rows | Columns | Class balance |
|----------|-----:|--------:|---------------|
| Unbalanced | 41,677 | 62 | 0: 37,756 · 1: 3,921 (~9.4% positive) |
| **SMOTE (D4-like)** | **75,512** | **61** | **0: 37,756 · 1: 37,756 (50/50)** |

Files:

- `data/interim/with_solubility_unbalanced.csv`  
- `data/processed/with_solubility_smote.csv`  
- `data/processed/dataset_meta.json`  

Paper Dataset 4 reported ~75,306 rows — our build is in the same range.

---

## Slide 12 — Label definition (important for slides)

From AID 504332 phenotype × activity outcome:

| Combination | Target |
|-------------|--------|
| **Inhibitor–Active** | **1** (active G9a inhibitor) |
| Inhibitor–Inconclusive | 0 |
| Inactive–Inactive | 0 |
| Activator–Inactive | 0 |

---

## Slide 13 — EDA highlights

**Command**

```bash
python scripts/eda_dataset.py
```

**Findings (SMOTE dataset)**

- No missing values  
- Classes perfectly balanced after SMOTE  
- 1 near-constant feature: `CBUC`  
- Stronger linear associations with target include:

| Feature | Corr. with target |
|---------|------------------:|
| RBC | −0.185 |
| S (sulphur mass %) | +0.168 |
| S_relative | +0.168 |
| Similarity | −0.098 |
| TPSA | +0.096 |
| N_relative | +0.095 |

**Insight aligned with the paper:** sulphur-related features stand out — paper’s feature-importance also highlighted sulphur chemistry for G9a inhibition.

**EDA outputs for the deck**

- `outputs/eda/*_eda_summary.md`  
- Figures: class balance, correlations, heatmaps, distributions, boxplots, pairplot  
  → `outputs/figures/eda/`

---

## Slide 14 — Paper’s final model (target to reproduce)

| Item | Value |
|------|--------|
| Algorithm | Random Forest (scikit-learn) |
| Features | `Volume_1`, `S_relative`, `S`, `N_relative`, `C_relative` |
| Hyperparams (paper-tuned) | `criterion=entropy`, `n_estimators=200`, `max_depth=20`, `min_samples_split=10` |
| Reported accuracy | ~**90%** (non-overfit; train–test gap ≤ 5%) |
| Train / test split | 80 / 20; scale **after** split |

Other models compared in paper: Decision Tree, Gradient Boosting, XGBoost, SVM; ANN (Optuna) and PySpark RF underperformed the scikit RF.

---

## Slide 15 — Training scripts ready (next demo steps)

Already implemented:

```bash
python scripts/04_compare_classifiers.py
python scripts/05_feature_importance.py
python scripts/06_train_final.py --sweep-depth
```

These write:

- Holdout + CV metric CSVs  
- Feature-importance tables and incremental-accuracy plot  
- Final model JSON + `outputs/models/final_rf.joblib`  

*(Full-data metrics should be regenerated after the complete dataset build — early smoke-test metrics used a tiny subset and are not paper-scale.)*

---

## Slide 16 — How to run (demo checklist)

```bash
cd impl
pip install -e .

python scripts/00_check_data.py
# (download already completed)
python scripts/create_dataset.py --balancing smote
python scripts/eda_dataset.py

python scripts/04_compare_classifiers.py
python scripts/05_feature_importance.py
python scripts/06_train_final.py --sweep-depth
```

Config: `configs/default.yaml` (solubility, balancing, final features, RF params).

---

## Slide 17 — Technology stack

- **Python 3.10+**  
- pandas, NumPy, SciPy  
- scikit-learn, imbalanced-learn (SMOTE / RUS)  
- XGBoost  
- RDKit (SMILES similarity)  
- ChemFormula (mass fractions)  
- matplotlib / seaborn (EDA & figures)  
- PyYAML, requests, tqdm, joblib  
- Optional (paper ANN): PyTorch, Optuna  

---

## Slide 18 — Contributions so far

1. Converted notebook research code into a **maintainable ML project**  
2. Closed the **missing-data gap** with a full PubChem coordinate download  
3. Produced a **D4-scale CSV** (~75.5k rows, 61 cols, balanced)  
4. Delivered **presentation-ready EDA** (tables + figures)  
5. Implemented the full **train → importance → final RF** script chain  
6. Documented config, steps, and reproducibility paths  

---

## Slide 19 — Limitations & honesty notes

- Original notebooks contain some quirks kept for fidelity (e.g. operator precedence in volume formulas)  
- Without-solubility datasets need additional raw files not shipped in the clone  
- Smoke-test runs earlier used limited downloads — **do not cite those accuracies** as paper reproduction  
- Full classifier comparison / final-model metrics on the new CSV should be refreshed for the presentation results slide  

---

## Slide 20 — Next steps (for remaining project work)

1. Run steps 04–06 on `with_solubility_smote.csv` and record metrics vs paper Table 1  
2. Overlay feature-importance ordering vs paper Figure 2 / 3  
3. Show overfitting curves (`max_depth` sweep) vs paper Figure 4  
4. Optional: RUS dataset (D3), ANN baseline, or without-solubility path  
5. Package results into final report / demo  

---

## Slide 21 — Key takeaways

- G9a inhibitor prediction is a **practical cheminformatics classification** problem on public PubChem data  
- **Solubility + SMOTE + Random Forest + 5 features** is the paper’s winning recipe (~90% accuracy)  
- This implementation makes that pipeline **runnable, inspectable, and presentable** as a real ML project  

---

## Appendix A — Quick fact sheet

| Item | Value |
|------|--------|
| Paper | Digital Discovery, 2024, 3, 2010–2018 |
| Bioassays | AID 504332 (G9a), AID 1996 (solubility) |
| Default experiment | With solubility + SMOTE (Dataset 4) |
| Unbalanced CSV | 41,677 rows · 3,921 actives |
| Balanced CSV | 75,512 rows · 50/50 |
| Final paper features | Volume_1, S_relative, S, N_relative, C_relative |
| Project root | `impl/` |

---

## Appendix B — Suggested slide visuals from our outputs

Use these files directly in the deck:

1. Class balance — `outputs/figures/eda/with_solubility_smote_class_balance.png`  
2. Corr with target — `outputs/figures/eda/with_solubility_smote_corr_with_target.png`  
3. Heatmap — `outputs/figures/eda/with_solubility_smote_corr_heatmap.png`  
4. Distributions — `outputs/figures/eda/with_solubility_smote_feature_distributions.png`  
5. Boxplots — `outputs/figures/eda/with_solubility_smote_boxplots_by_class.png`  
6. Pairplot — `outputs/figures/eda/with_solubility_smote_pairplot.png`  
7. Methodology diagram — redraw Slide 5 flowchart in PowerPoint / draw.io  

---

## Appendix C — Citation

> Mariya L. Ivanova, Nicola Russo, Nadia Djaid and Konstantin Nikolic,  
> *Application of machine learning for predicting G9a inhibitors*,  
> Digital Discovery, 2024, **3**, 2010–2018.  
> https://doi.org/10.1039/d4dd00101j  

Original code: https://github.com/articlesmli/G9a_clsf  

---

*Document generated for presentation use — reflects project status after dataset creation and EDA completion.*
