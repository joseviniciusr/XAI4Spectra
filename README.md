# XAI4Spectra — Repository Map & Architecture Guide

> **Purpose:** Explainability methods for spectral data (XRF and Vis-NIR), with a focus on binary classification tasks applied to real-world analytical chemistry datasets. The core contribution is **SMeX** (Spectral Method of Explainability), a novel XAI technique that competes against SHAP, VIP (Variable Importance in Projection), and permutation/perturbation importance baselines.

---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Core Library Files](#2-core-library-files)
3. [Datasets](#3-datasets)
4. [Experiment Directories](#4-experiment-directories)
5. [Result Aggregation](#5-result-aggregation)
6. [DPG Submodule](#6-dpg-submodule)
7. [Root-Level Notebooks](#7-root-level-notebooks)
8. [Semantic Groupings & Refactoring Opportunities](#8-semantic-groupings--refactoring-opportunities)
9. [Metrics & Comparison Pipeline](#9-metrics--comparison-pipeline)
10. [End-to-End Workflow](#10-end-to-end-workflow)

---

## 1. Project Overview

The project evaluates how well different XAI techniques agree on **which spectral zones matter** for distinguishing two classes (A vs B) in analytical spectroscopy data. The comparison is quantified using **RBO (Rank-Biased Overlap)**, a similarity metric for ranked lists.

### What is SMeX?

SMeX is the novel technique developed here. At a high level it:

1. Segments the spectrum into **named zones** (e.g., `Ca ka`, `Fe ka`, etc.)
2. Aggregates each zone to a scalar per sample (sum / mean / median / max)
3. Generates **predicates** — thresholded rules like `Ca ka <= 25.5` — via quantile cuts
4. Runs **bagging** or **KS-based cross-validation folds** to build bags of predicates
5. Scores each predicate via **Mutual Information** or **Covariance** with model predictions
6. Constructs a **directed graph** of predicates, weighted by co-occurrence and importance scores
7. Computes **Local Reaching Centrality (LRC)** on each node — the LRC ranking of zones is the explanation
8. Uses **Genetic Algorithm (GA)** to optimise the internal parameters of SMeX

The LRC-ranked zone list is then compared to VIP / SHAP / permutation importance rankings via RBO to measure alignment.

---

## 2. Core Library Files

These files are the shared backbone of the entire project. All experiment notebooks import from them.

### `modeling.py`
Model fitting and evaluation for all three classifiers. Contains three public functions:

| Function | Model | `aim=` |
|---|---|---|
| `pls_optimized()` | PLS-DA / PLS-R | `'classification'` / `'regression'` |
| `svm_optimized()` | SVC / SVR | `'classification'` / `'regression'` |
| `mlp_optimized()` | MLPClassifier / MLPRegressor | `'classification'` / `'regression'` |

All three follow the same contract: accept `(Xcal, ycal, Xpred, ypred, aim, ...)`, return `(df_results, calres, predres, model, ...)`. Each computes calibration and prediction metrics including R², RMSE, RPD, RPIQ, bias, accuracy, sensitivity, and specificity.

Helper functions inside `modeling.py`:
- `vip_scores(pls_model)` — Variable Importance in Projection from a fitted PLS model
- `explained_variance_from_scores()` — Percent variance explained for X and Y per component

### `explaining.py`
The SMeX engine. Key functions in pipeline order:

| Step | Function | Output |
|---|---|---|
| 1 | `extract_spectral_zones(Xcal, cuts)` | Dict of zone DataFrames |
| 2 | `aggregate_spectral_zones(zones, aggregator)` | DataFrame of zone scalars per sample |
| 3 | `predicates_by_quantiles(zone_sums_df, quantiles)` | predicates_df, indicator matrix, co-occurrence matrix |
| 4 | `create_predicate_info_dict(...)` | Dict mapping rule → DataFrame of samples |
| 5a | `bagging_predicates(...)` | Bags dict (random subsampling strategy) |
| 5b | `kfold_predicates_roundrobin(...)` *(in ks_folding.py)* | Fold dict (deterministic KS strategy) |
| 6 | `calculate_predicate_metrics(bags, metric, threshold)` | Per-bag MI or covariance rankings |
| 7 | `build_predicate_graph(...)` / `build_fold_predicate_graph(...)` | `nx.DiGraph` |
| 8 | `calculate_lrc_single_graph(graph, predicates_df)` | LRC DataFrame ranked by importance |
| — | `calculate_predicate_ranking_mean(mi_results_dict)` | Averaged ranking across folds/bags |

Additional functions:
- `calculate_lrc(graphs_by_seed, predicates_df)` — multi-graph (multi-seed) LRC variant
- `spectral_perturbation_importance(model, X, ...)` — perturbation baseline inside the SMeX pipeline

### `preprocessings.py`
Spectral preprocessing methods, all returning the preprocessed matrix and parameters needed to apply the same transform to the prediction set:

| Function | Method |
|---|---|
| `poisson(X, mc=True)` | Poisson scaling (÷ √mean) + optional mean centering |
| `modified_poisson(X, degree)` | Generalised Poisson scaling |
| `pareto(X, mc=True)` | Pareto scaling (÷ √std) |
| `mc(X)` | Mean centering only |
| `auto_scaling(X)` | Z-score (÷ std then − mean) |
| `msc(X, reference)` | Multiplicative Scatter Correction |

### `ks_folding.py`
Cross-validation strategy based on Kennard-Stone (KS) ordering + round-robin assignment. Provides two strategies to split predicate data for SMeX evaluation:

- **Global strategy** (`per_predicate=False`): KS applied once to all samples; all predicates share the same folds
- **Per-predicate strategy** (`per_predicate=True`): KS applied independently per predicate; may keep more predicates alive

Key functions:
- `ks_ordered_indices(X)` — multidimensional KS ordering
- `ks_ordered_indices_1d(values)` — 1-D KS ordering (wraps values into 2-D dummy)
- `ks_roundrobin_kfold(X, k)` — produces `k` balanced, diverse folds
- `kfold_predicates_roundrobin(...)` — main entry-point: builds `{'Fold_N': {rule: df}}` structure

### `permutation.py`
Permutation importance and spectral perturbation for predicates. Provides model-agnostic XAI baselines and supports both classification and regression tasks:

- `calculate_predicate_metrics_permutation(...)` — permutation importance (shuffles zone columns) per fold
- `calculate_predicate_perturbation(...)` — perturbation importance (zeroes zone columns) per fold
- `spectral_perturbation_importance(model, X, ...)` — global (non-fold-based) perturbation importance
- `get_zone_columns_from_predicate(...)` — utility: maps a predicate rule → list of spectral columns

> **Note:** `permutation.py` and the analogous block inside `explaining.py` contain duplicated code. The `permutation.py` file appears to be the extracted, standalone version.

### `GA_otimization.py` / `GA_otimization_parallel.py`
Genetic Algorithm (DEAP-based) that optimises the 6 internal hyperparameters of SMeX:

| Gene | Choices / Range |
|---|---|
| Aggregator function | `sum`, `median`, `max` |
| Association metric | `mutual_info`, `covariance` |
| Number of bags | 20 – 150 |
| Sample fraction per bag | 0.50 – 0.90 |
| Min. samples per predicate fraction | 0.05 – 0.30 |
| Replacement (bootstrap) | `True` / `False` |

**Fitness function:** RBO similarity between the LRC-ranked zone list produced by SMeX and the VIP-ranked zone list from PLS-DA. Runs over multiple random seeds and saves HOF (Hall of Fame) and statistics to CSV.

`GA_otimization_parallel.py` is identical except it uses `multiprocessing` for parallel fitness evaluation of individuals.

### `synthetic.py`
Generator of synthetic spectral data using Gaussian peaks. Produces two-class datasets where classes differ in which peaks they contain, enabling ground-truth evaluation of XAI techniques.

Key functions:
- `modelo_pico_gaussiano(x, centro, amplitude, largura)` — single Gaussian peak
- `_gerar_espectro_unico(...)` — single spectrum with variable-amplitude peaks + noise
- `generate_synthetic_spectral_data(config, n_pontos, ...)` — full dataset generator

---

## 3. Datasets

### `XRF_databases/` — X-Ray Fluorescence real datasets
Each sub-directory contains a `plsda/` folder with the raw CSV for binary classification:

| Dataset | Application domain |
|---|---|
| `soil/` | Soil type discrimination via XRF |
| `bank_notes/` | Authentic vs. counterfeit bank note detection |
| `forage/` | Forage quality / variety classification |
| `milk/` | Milk adulteration or type detection |
| `soil_types/` | Multi-class soil type (run as binary pairs) |
| `ecigar/` | Electronic cigarette / tobacco classification |

### `VNIR_databases/soil/` — Vis-NIR reflectance
A single soil dataset measured in the visible-near-infrared range, used to test SMeX with a sensor modality different from XRF.

### `Synthetic_databases/`
- `plsda/` — generated synthetic XRF-like data
- `synthetic_generator.ipynb` — interactive notebook for tuning peak configurations

---

## 4. Experiment Directories

Each experiment directory follows the same pattern: one sub-folder per dataset, containing a Jupyter notebook and result files. The notebooks are essentially identical except for dataset-specific parameters (spectral cuts, LV count, data path, preprocessing).

### `PLS/` — Experiments run with PLS-DA as the predictive model
```
PLS/{dataset}/
  ├── {dataset}_notebook_full_comparisson.ipynb  ← main experiment notebook
  ├── feature_importance.csv                     ← raw per-method importance scores
  ├── lrc_cov_natural.csv                        ← SMeX LRC scores (covariance metric)
  ├── lrc_pert_natural.csv                       ← SMeX LRC scores (perturbation metric)
  ├── rbo_rank.csv                               ← RBO similarity vs. VIP / SHAP / permutation
  ├── shap_{dataset}.csv                         ← SHAP values
  └── shap_{dataset}.py                          ← script that produced SHAP values
```
Datasets covered: `bank_notes`, `forage`, `milk`, `soil`, `soil_types`, `synthetic`

### `SVM/` — Experiments run with SVM (SVC) as the predictive model
Same structure as `PLS/`. Additionally contains an `ecigar/` sub-directory (only present here).

### `MLP/` — Experiments run with MLP (MLPClassifier) as the predictive model
Same structure as `PLS/` and `SVM/`.

### `Graph_v2/` — Variant of the SMeX graph with a different edge-weight strategy
```
Graph_v2/{dataset}/
  ├── {dataset}_notebook_full_comparisson.ipynb
  ├── feature_importance.csv
  ├── Perturbation_method.xlsx
  ├── rbo_rank.csv
  └── shap_{dataset}.csv / shap_{dataset}.py
```
The notebooks here use a revised `build_predicate_graph` variant (possibly `build_fold_predicate_graph` with `weight_mode='cooccurrence'`), exploring whether different weighting schemes change the explanation quality.

### `KS_folding_tests/` — Cross-validation strategy comparison
Tests the KS + round-robin folding strategy against simple bagging. Includes a `theory.ipynb` notebook and per-dataset results. Also includes a `Permutation_tests/ks_folding.py` copy for reproducibility.

### `Permutation_tests/` — Permutation importance as an XAI baseline
```
Permutation_tests/{dataset}/
  ├── {dataset}_notebook.ipynb
  ├── features_importance_{dataset}.xlsx
  ├── rbo_{dataset}.xlsx
  └── shap_{dataset}.csv
```
Additionally has a `vnir/` sub-directory for the VNIR soil dataset, and a `ks_folding.py` local copy.

### `Perturbation/` — Spectral perturbation importance as an XAI baseline
```
Perturbation/{dataset}/
  ├── {dataset}_notebook_full_comparisson.ipynb
  ├── feature_importance.csv
  ├── Perturbation_method.xlsx / Permutation_method.xlsx
  ├── rbo_rank.csv
  ├── shap_{dataset}.csv / shap_{dataset}.py
  └── permutation.py  ← local copy for self-contained reproduction
```

### `PCA_aggregator/` — PCA as the zone aggregator instead of sum/mean/etc.
Tests a PCA-based zone representation inside the SMeX pipeline:
```
PCA_aggregator/
  ├── MLP/{dataset}/
  ├── PLS/{dataset}/
  └── SVM/{dataset}/
```
Each leaf contains the same result files as the main experiment directories. Referenced in `summary/analisys.ipynb` as `rbo_pca_results`.

---

## 5. Result Aggregation

### `summary/`
Central place where results from all experiments are merged and compared.

- `analisys.ipynb` — loads `rbo_rank.csv` from every `{model}/{dataset}/` directory and `PCA_aggregator/{model}/{dataset}/` directory. Produces comparison matrices and visualisations showing how well SMeX aligns with VIP, SHAP, and permutation baselines across all datasets and models.
- `rbo_consolidated_{dataset}.csv` — consolidated RBO scores per dataset (all models merged)
- `rbo_lrc_comparison_{dataset}.csv` — direct LRC vs VIP/SHAP comparison per dataset

---


### `DPG/`
A separate sub-project (own `pyproject.toml`, `requirements.txt`, `LICENSE`) implementing **Data-Predictive Graphs** — a different graph-based XAI technique. It is used as a point of comparison and inspiration.

Key files:
- `dpg/core.py` — graph construction core
- `dpg/sklearn_dpg.py` — sklearn-compatible wrapper
- `dpg/visualizer.py` — graph visualisation
- `dpg/utils.py` — utilities
- `example_dpg_notebook.ipynb`, `iris_dpg_notebook.ipynb` — usage examples
- `dpg_aug.ipynb`, `analyze.ipynb`, `analyze_mean.ipynb` — experimental analyses

---

## 7. Root-Level Notebooks

| Notebook | Purpose |
|---|---|
| `initial_notebook.ipynb` | First exploration / proof-of-concept of the SMeX pipeline |
| `pseudocode.ipynb` | Step-by-step pseudocode of the SMeX algorithm, used for documentation |
| `pseudocode.tex` | LaTeX version of the pseudocode (for paper) |
| `pseudocode.py` | Runnable Python version of the pseudocode (mirrors `pseudocode.ipynb`) |
| `pca_aggregator.ipynb` | Prototype of PCA-based zone aggregation |
| `pca_aggregator_implementation.ipynb` | Full implementation / validation of PCA aggregator |
| `bank_notes_notebook.ipynb` | Early standalone experiment on bank-notes dataset |
| `soil_notebook.ipynb` | Early standalone experiment on soil dataset |
| `soil_regression_notebook.ipynb` | Regression variant: predicting a continuous soil property |
| `soil_types_notebook.ipynb` | Multi-class soil types experiment |
| `synthetic_notebook.ipynb` | Full SMeX pipeline on the synthetic dataset |
| `ecigar_notebook copy.ipynb` | Experiment on e-cigarette dataset |
| `tests.ipynb` | Scratch tests / debugging |
| `DPG_comparing.ipynb` | Side-by-side comparison of SMeX vs DPG results |
| `summarizing_GA_results.ipynb` | Loads GA HOF CSVs and surfaces the best parameters found |

---

## 8. Semantic Groupings & Refactoring Opportunities

### Group A — Dataset-specific experiment notebooks (nearly identical)
The following notebooks share the exact same structure and differ only in:
- Dataset path and CSV filename
- Spectral boundary (`spectral_range`)
- Spectral zone cut definitions (`spectral_cuts`)
- Number of latent variables (`LV`)
- Preprocessing method (Poisson for XRF, MC for synthetic/VNIR)

**Notebooks in this group:**
```
PLS/bank_notes/bank_notes_notebook_full_comparisson.ipynb
PLS/soil/soil_notebook_full_comparisson.ipynb
PLS/forage/forage_notebook_full_comparisson.ipynb
PLS/milk/milk_notebook_full_comparisson.ipynb
PLS/soil_types/...
PLS/synthetic/...
SVM/{same six datasets}/...
MLP/{same six datasets}/...
Graph_v2/{same datasets}/...
KS_folding_tests/{datasets}/...
Permutation_tests/{datasets}/...
Perturbation/{datasets}/...
PCA_aggregator/MLP|PLS|SVM/{datasets}/...
```

**Refactoring suggestion:** Extract a single parameterised `run_experiment(config)` function or script (similar to what `GA_otimization.py` already does with its `instructions` dictionary), and run it once per dataset/model combination. The `instructions` dict pattern in `GA_otimization.py` is exactly the right approach — it just needs to be applied to all experiment notebooks.

### Group B — Importance calculation variants (same algorithm, different metric)
The files `lrc_cov_natural.csv` and `lrc_pert_natural.csv` in each result directory correspond to:
- **Covariance-metric SMeX** — uses `calculate_predicate_metrics(..., metric='covariance')`
- **Perturbation-metric SMeX** — uses `calculate_predicate_perturbation(...)` or `calculate_predicate_metrics_permutation(...)`

The code paths that produce these are parallel and could be unified into a single `run_smex(metric)` call.

### Group C — `permutation.py` vs. `explaining.py` duplication
`permutation.py` (root level) contains `calculate_predicate_metrics_permutation`, `calculate_predicate_perturbation`, `get_zone_columns_from_predicate`, and `_manual_block_permutation`.

The same functions also exist inside `explaining.py` (lines ~1900+). The root-level `permutation.py` is the clean extracted version; `explaining.py` should drop its copies and import from `permutation.py`.

Similarly, each `Perturbation/{dataset}/permutation.py` is a verbatim copy of the root `permutation.py`, kept for self-contained notebook execution. These copies can be replaced by a `sys.path` parent-import (`import permutation`).

### Group D — SHAP scripts (`shap_{dataset}.py`)
Each result directory contains a small Python script that loads data, trains a model, and computes SHAP values, saving them to `shap_{dataset}.csv`. All these scripts are identical apart from dataset path, spectral range, and LV count. They should be merged into a single `compute_shap.py --dataset soil --model PLS` CLI script.

### Group E — `GA_otimization.py` and `GA_otimization_parallel.py`
These two files are identical except for the `multiprocessing` pool setup. The shared implementation should live in one file with a `parallel=True/False` flag.

---

## 9. Metrics & Comparison Pipeline

### XAI method outputs produced per experiment

| Method | Output | File |
|---|---|---|
| VIP (PLS only) | Ranked zone list | derived inside notebook |
| SHAP | Per-feature importance → aggregated to zones | `shap_{dataset}.csv` |
| SMeX (covariance) | LRC-ranked zone list | `lrc_cov_natural.csv` |
| SMeX (perturbation) | LRC-ranked zone list | `lrc_pert_natural.csv` |
| Permutation importance | Per-zone importance | `feature_importance.csv` (partial) |
| Spectral perturbation | Per-zone importance | `Perturbation_method.xlsx` |

### Comparison metric: RBO (Rank-Biased Overlap)
All rankings are compared pairwise using `rbo.RankingSimilarity(...).rbo(p=0.7, k=10)`:
- `p=0.7` weights top-ranked elements more heavily
- `k=10` caps comparison at top 10 zones
- Higher RBO → the two methods agree more on which zones matter

Results are stored in `rbo_rank.csv` (one row per method-pair) and consolidated in `summary/`.

### `summary/analisys.ipynb` workflow
1. Loads `rbo_rank.csv` from every `{Model}/{dataset}/` and `PCA_aggregator/{Model}/{dataset}/` path
2. Concatenates into a master DataFrame
3. Produces co-occurrence / agreement matrices across models and datasets
4. The CSVs `rbo_consolidated_{dataset}.csv` and `rbo_lrc_comparison_{dataset}.csv` are pre-generated summaries

---

## 10. End-to-End Workflow

```
Raw Spectrum (CSV)
       │
       ▼
preprocessings.py  ←── Poisson / Pareto / MC / MSC
       │
       ▼
modeling.py  ←──────── pls_optimized / svm_optimized / mlp_optimized
 (Xcal → model, VIP)
       │
       ├─────────────────────────────────────────────────────────────┐
       │ SMeX path                                                   │ Baseline paths
       ▼                                                             ▼
explaining.py                                               SHAP / Permutation /
  extract_spectral_zones  →  aggregate_spectral_zones         Perturbation
  predicates_by_quantiles
  bagging_predicates  (or)  ks_folding.py:kfold_predicates_roundrobin
  calculate_predicate_metrics  (MI / Cov)
    (or)  permutation.py:calculate_predicate_metrics_permutation
  build_fold_predicate_graph
  calculate_lrc_single_graph
       │
       ▼
  LRC-ranked zone list
       │
       ▼
  rbo.RankingSimilarity vs VIP / SHAP / permutation
       │
       ▼
  rbo_rank.csv   ──►  summary/analisys.ipynb  ──►  rbo_consolidated_*.csv
```

**GA optimisation loop (GA_otimization.py):**
The fitness function wraps the entire SMeX path above and uses RBO vs VIP as the fitness score. DEAP evolves the 6 SMeX hyperparameters over 50 generations to maximise agreement with VIP.
