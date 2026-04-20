# AML-Node16-Challenge

**Decentralized AI for Anti-Money Laundering (AML)**

A privacy-preserving, federated machine learning system for detecting money laundering in financial transaction data — without sharing raw data between institutions.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Code Reference](#code-reference)
  - [ResultLogger](#resultlogger)
  - [AMLModel](#amlmodel)
- [Training Pipeline (5 Phases)](#training-pipeline-5-phases)
- [Federated Learning Design](#federated-learning-design)
- [Evaluation Metrics](#evaluation-metrics)
- [Output Files](#output-files)
- [Dependencies](#dependencies)

---

## Overview

This project tackles the detection of money laundering in financial transactions using a **federated learning** approach. Instead of centralising sensitive transaction data from multiple banks, each bank trains a model locally. Models are then combined sequentially using LightGBM's `init_model` parameter, so that knowledge is shared without exposing raw data.

Key design decisions:

- **Privacy-first**: No raw data is ever shared between banks.
- **Temporal splits**: Data is split chronologically, not randomly, to prevent data leakage and reflect real-world deployment.
- **LightGBM with native categoricals**: Categorical features (e.g. transaction type, currency) are handled natively — no manual encoding.
- **PR-AUC as primary metric**: Given extreme class imbalance (laundering is rare), Precision-Recall AUC is a more meaningful metric than accuracy or ROC-AUC.
- **Automatic class imbalance handling**: `is_unbalance=True` is set in LightGBM by default.

---

## Architecture

```
┌───────────────────────────────────────────────────────┐
│                   Federated Pipeline                  │
│                                                       │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────┐ │
│  │  Large Bank │ → │  Medium Bank │ → │ Small Bank │ │
│  │  (trains    │   │ (continues   │   │(continues  │ │
│  │   first)    │   │  training)   │   │ training)  │ │
│  └─────────────┘   └──────────────┘   └────────────┘ │
│         ↓                                    ↓        │
│   Hyperparameter                     Global Federated │
│      Tuning                               Model       │
└───────────────────────────────────────────────────────┘
```

The federated loop:
1. Bank A trains a LightGBM model on its local data.
2. The trained model (not the data) is passed to Bank B via `init_model`.
3. Bank B continues training on its own local data.
4. This continues for all banks.
5. The final model has learned patterns from every bank without any data sharing.

---

## Repository Structure

```
AML-Node16-challenge/
│
├── run_AML.py                          # Main script — full federated pipeline
├── EDA & Preprocessing.ipynb          # Data exploration and cleaning notebook
├── requirements.txt                   # Python dependencies
├── results.txt                        # Auto-generated training logs & metrics
├── best_global_federated_model.pkl    # Saved global federated model
├── Synopsis_ AML Node16 Challenge.pdf # Project synopsis document
│
└── clean_data/                        # Preprocessed data (Parquet format)
    ├── large_bank_HI_transactions_preprocessed.parquet
    ├── medium_bank_LI_transactions_preprocessed.parquet
    └── small_bank_HI_transactions_preprocessed.parquet
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/sm-ak-r33/AML-Node16-challenge.git
cd AML-Node16-challenge
```

**2. Create and activate a Conda environment**

```bash
conda create -n aml-env python=3.9 -y
conda activate aml-env
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. (Optional) Enable UTF-8 encoding on Windows**

```bash
export PYTHONUTF8=1
```

---

## Data

### Source

The raw transaction data comes from the IBM AML dataset on Kaggle:

> [IBM Transactions for Anti-Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data?select=HI-Large_Trans.csv)

Download `HI-Large_Trans.csv` and run the `EDA & Preprocessing.ipynb` notebook to generate the cleaned Parquet files, or use the pre-cleaned files already provided in `clean_data/`.

### Preprocessing Steps (EDA & Preprocessing.ipynb)

- Duplicate removal
- Timestamp parsing and extraction of temporal features (`Hour`, `Day`, `DayOfWeek`, `Month`, `Year`)
- Categorical encoding to `category` dtype for LightGBM compatibility
- Data export to Parquet format using `fastparquet` for performance and scalability

### Data Split

Each bank's data is split **temporally** (chronologically), preserving the time-ordered nature of financial transactions:

| Split      | Proportion | Purpose                        |
|------------|-----------|--------------------------------|
| Train      | 60%       | Model training                 |
| Validation | 20%       | Hyperparameter tuning          |
| Test       | 20%       | Final evaluation               |

---

## Usage

Run the full federated pipeline end-to-end:

```bash
python run_AML.py
```

This executes all 5 phases automatically and writes results to `results.txt`.

---

## Code Reference

### `ResultLogger`

Handles dual-output logging — writes to both the console and `results.txt` simultaneously.

```python
logger = ResultLogger(filepath='results.txt')
logger.log("Training started...")
logger.close()
```

| Method | Description |
|--------|-------------|
| `__init__(filepath)` | Opens the output file for writing |
| `log(message, to_console=True)` | Writes a message to the file (and optionally the console) |
| `close()` | Flushes and closes the file handle |

---

### `AMLModel`

The core class encapsulating all model logic: data loading, splitting, training, evaluation, grid search, saving/loading, and federated coordination.

#### Constructor

```python
model = AMLModel(
    categorical_features=['Payment_Currency', 'Receiving_Currency', 'Payment_Format'],
    random_state=42,
    verbose=False,
    bank_name="Large_Bank"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `categorical_features` | `list` | `[]` | Column names to treat as categorical |
| `random_state` | `int` | `42` | Random seed for reproducibility |
| `verbose` | `bool` | `False` | Whether to show detailed LightGBM logs |
| `bank_name` | `str` | `"Unknown Bank"` | Identifier for this institution |

#### Instance Methods

| Method | Description |
|--------|-------------|
| `load_data(filepath, logger)` | Loads a Parquet file, removes duplicates, parses timestamps, detects categoricals |
| `temporal_split(train_ratio, val_ratio, test_ratio, logger)` | Splits data chronologically into train/val/test sets |
| `build_model(lgb_params)` | Builds a `LGBMClassifier` with given parameters (defaults to `is_unbalance=True`) |
| `train(init_model)` | Trains the model; if `init_model` is provided, continues training from it (federated) |
| `evaluate(X, y, dataset_name)` | Returns a dict of metrics: PR-AUC, ROC-AUC, accuracy, precision, recall, F1 |
| `local_grid_search(param_grid, logger)` | Grid search over hyperparameters using PR-AUC on the local validation set |
| `train_final_local_model(logger)` | Trains the final model using `best_params` found by grid search |
| `save_model(filepath, logger)` | Serialises the trained model and metadata to a `.pkl` file via `joblib` |
| `load_model(filepath, logger)` | Loads a previously saved model from a `.pkl` file |

#### Static Methods

| Method | Description |
|--------|-------------|
| `federated_training_sequential(bank_models, best_params, logger)` | Trains the global federated model by sequentially passing `init_model` across all banks |
| `compare_local_vs_global(bank_models, global_model, logger)` | Evaluates and compares local vs global model performance on each bank's test set |
| `setup_federated_banks(filepaths, bank_names, categorical_features, ...)` | Convenience factory: loads and splits data for multiple banks in one call |

---

## Training Pipeline (5 Phases)

### Phase 1 — Setup Banks

Each bank's data is loaded from Parquet, deduplicated, temporally split, and made ready for training. All object columns are auto-detected as categorical.

### Phase 2 — Hyperparameter Tuning (Largest Bank Only)

Grid search is run only on the largest bank's local data to save computation. The default search space is:

```python
param_grid = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100],
    'num_leaves': [31, 50]
}
```

The combination with the highest **Validation PR-AUC** is selected as `best_params`.

### Phase 3 — Local Model Training

Each bank trains a final local LightGBM model using `best_params` from Phase 2. These serve as baselines for the comparison in Phase 5.

### Phase 4 — Federated Training (Sequential)

The global model is built by passing the model weights from bank to bank:

```
Bank 1 trains → model_1
Bank 2 trains with init_model=model_1 → model_2
Bank 3 trains with init_model=model_2 → model_3 (global model)
```

No raw data is shared at any point. The final global model is saved to `best_global_federated_model.pkl`.

### Phase 5 — Comparison

The global federated model and each bank's local model are both evaluated on every bank's local test set. A summary table reports improvements in PR-AUC and F1 score.

---

## Federated Learning Design

This project implements **horizontal federated learning**, where all banks share the same feature schema but train on different sets of transactions.

The approach uses LightGBM's `init_model` parameter for sequential knowledge transfer. This is a form of **model warm-starting** rather than traditional federated averaging (FedAvg). Key properties:

- **No gradient sharing** — the full model is passed between steps.
- **Order matters** — banks are trained in sequence; the largest bank goes first.
- **No data pooling** — raw transactions never leave the institution.
- **Consistent hyperparameters** — all banks use the same `best_params` to ensure compatibility.

---

## Evaluation Metrics

| Metric | Notes |
|--------|-------|
| **PR-AUC** *(primary)* | Precision-Recall AUC; best for imbalanced classification |
| ROC-AUC | Area under ROC curve |
| F1 Score | Harmonic mean of precision and recall |
| Precision | Of all flagged transactions, how many are truly laundering |
| Recall | Of all laundering transactions, how many were caught |
| Accuracy | Overall correctness (can be misleading with class imbalance) |

PR-AUC is used as the **primary optimisation target** during grid search and comparison, because the dataset is heavily imbalanced (laundering transactions are a small fraction of all transactions).

---

## Output Files

| File | Description |
|------|-------------|
| `results.txt` | Full training log including metrics for all phases |
| `best_global_federated_model.pkl` | Serialised global federated model (LightGBM + metadata) |

The `.pkl` file stores:
```python
{
    'model': lgb.LGBMClassifier,
    'categorical_features': list,
    'best_params': dict,
    'bank_name': str
}
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥ 2.0 | Data loading and manipulation |
| `numpy` | ≥ 1.26 | Numerical operations |
| `scikit-learn` | ≥ 1.3 | Metrics and grid search utilities |
| `lightgbm` | ≥ 4.0 | Gradient boosted tree model with native categorical support |
| `matplotlib` | ≥ 3.8 | Visualisation (optional) |
| `seaborn` | ≥ 0.12 | Visualisation (optional) |
| `fastparquet` | latest | Efficient Parquet file I/O |
| `tabulate` | latest | Formatted table output in logs |
| `joblib` | (via sklearn) | Model serialisation |

---

## License

This project is part of the AML Node16 Challenge. See the [Synopsis PDF](Synopsis_%20AML%20Node16%20Challenge.pdf) for full challenge context.
