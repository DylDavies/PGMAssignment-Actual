# Temporal and Generative PGMs for Zero-Day Anomaly Detection in SCADA Systems

**Dylan Davies — Student Number: 2672598**

---

## Overview

This project implements and compares three Probabilistic Graphical Model architectures for anomaly detection on the Secure Water Treatment (SWaT) dataset:

- **Supervised HMM** (`shmm`) — BN-augmented HMM fitted on labelled training data
- **Generative One-Class BN** (`goc`) — One-class Bayesian Network trained exclusively on normal data
- **Unsupervised HMM** (`hmm`) — Per-sensor CategoricalHMM trained via Baum-Welch (EM) on normal data

---

## Requirements

Python 3.11 was used for this project.

Install all dependencies with:

```bash
pip install -r requirements.txt
```

> **Note:** `pgmpy` is pinned to `0.1.25`. Using a different version may cause API incompatibilities.

### Kaggle API Setup

The dataset is downloaded automatically via the Kaggle API on first run. You must have a Kaggle account and a configured API token:

1. Go to [kaggle.com](https://www.kaggle.com) → Account (Top Right) → Your API Tokens → Generate New Token
2. Follow the instructions on the website after entering a name and creating the token.

The dataset (`vishala28/swat-dataset-secure-water-treatment-system`) will be downloaded to `data/` automatically. Subsequent runs will skip the download if the files are already present. You can also download the dataset manually from [here](https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system) and place the CSV files in the `data/` directory (create it if it isn't there - the download script will create the directory automatically if it isn't there before downloading).

---

## Running the Code

```bash
python main.py
```

All output (confusion matrix plots, BN structure plots) is saved to the `output/` directory.

### Selecting Which Models to Run

Open `config.py` and edit the `MODELS_TO_RUN` list at the bottom of the file:

```python
MODELS_TO_RUN: list[Literal['shmm', 'goc', 'hmm']] = ['goc']
```

| Value | Model |
|---|---|
| `'goc'` | Generative One-Class Bayesian Network |
| `'shmm'` | Supervised HMM |
| `'hmm'` | Unsupervised HMM (Baum-Welch) |

Any combination is valid, e.g. `['shmm', 'goc', 'hmm']` to run all three.

> **Important — Timing Accuracy:** If you wish to reproduce the training time reported for the Generative BN in Table IV of the paper (~68s), run it in isolation: `MODELS_TO_RUN = ['goc']`. Because structure learning is shared between `shmm` and `goc`, running `shmm` first will cache the structures, and the reported GOC time will include that shared cost if both are included together.

---

## Reproducing Paper Results

The key hyperparameters from Tables II and III in the paper are controlled entirely from `config.py`:

```python
N_BINS = 4                          # Discretisation granularity (Table II: 3, 4, 5, 6)
THRESHOLD_PERC = 1                  # Anomaly threshold percentile (Table III: 0.5, 1.0, 2.0, 5.0, 10.0)
N_COMPONENTS = 3                    # Number of latent HMM states
DISCRETISE_STRATEGY = "uniform"     # Binning strategy ('uniform', 'quantile', 'kmeans')
```

To reproduce a specific row from the sensitivity tables, change the relevant value and re-run `python main.py`.

> **Note on caching:** Structure learning results are cached to `output/cache/` via `joblib`. If you change `N_BINS` or `DISCRETISE_STRATEGY`, delete the cache directory (or its contents) before re-running, otherwise the old structures will be reused.

---

## Code Structure

```
.
├── main.py                          # Entry point — orchestrates the full pipeline
├── config.py                        # All hyperparameters and path configuration
├── requirements.txt
│
├── helpers/
│   ├── data_helper.py               # Kaggle download, CSV loading, train/test split
│   ├── preprocessing_helper.py      # Imputation, discretisation, zero-variance filtering
│   ├── structure_learning_helper.py # BIC-penalised Hill Climb Search (cached)
│   ├── parameter_learning_helper.py # BDeu parameter fitting; transition matrix learning
│   ├── shmm_inference_helper.py     # Supervised HMM forward-algorithm inference
│   ├── generative_inference_helper.py # One-class BN log-likelihood thresholding
│   ├── hmm_inference_helper.py      # Per-sensor Baum-Welch HMM training and inference
│   └── evaluation_helper.py        # Classification report and confusion matrix plotting
│
├── data/                            # Downloaded SWaT CSVs (created on first run)
└── output/                          # Saved plots and joblib cache
    ├── BNPlots.png                  # Learned DAG structures (Fig. 1 in paper)
    ├── confusion_matrix_*.png       # Confusion matrices (Figs. 2–4 in paper)
    └── cache/                       # joblib cache — delete to force recomputation
```

### Pipeline Flow

```
data_helper        →  Load & merge SWaT CSVs, chronological train/test split (80/20)
preprocessing      →  Median/mode imputation, zero-variance drop, uniform discretisation
structure_learning →  Per-stage BIC Hill Climb on normal training data (cached)
parameter_learning →  BDeu Bayesian estimation of CPDs; transition matrix from labels
inference          →  Forward algorithm (SHMM) / log-likelihood threshold (GOC, HMM)
evaluation         →  Per-class precision, recall, F1; confusion matrix plots
```

---

## Expected Runtimes

Approximate training times on a standard laptop (as reported in Table IV):

| Model | Approx. Training Time |
|---|---|
| Generative BN | ~68 seconds |
| Supervised HMM | ~17 minutes |
| Unsupervised HMM (Baum-Welch) | ~38 minutes |
