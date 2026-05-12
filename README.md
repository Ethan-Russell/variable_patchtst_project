# VariablePatchTST

This repository contains the model and data handling code used for the VariablePatchTST report for Cornell University course CS 6787 Advanced Machine Learning Systems.

## Introduction

VariablePatchTST is a dynamically patching algorithm that utilizes a multi-resolution hierarchy of temporal patches that increase in length according to a geometric progression. Patch lengths start with a minimum size times powers of two, with more tokens allocated to finer scales so that recent timesteps are represented at higher resolution and older timesteps at coarser resolution.

## What is implemented

- **`VariablePatchAlgorithm`** (`get_perfect_patches`): builds a fixed number of patches that exactly cover the context window, with scale-specific patch sizes and a controllable overlap toward the recent end of the series.
- **`UniformPatchAlgorithm`**: standard fixed patch size and stride (baseline).
- **`SimplePatchTST`**: splits each channel into patches, projects heterogeneous patch lengths to a shared `d_model` via **`DynamicEmbedding`**, adds positional encoding, runs a PyTorch **`TransformerEncoder`**, and predicts the forecast horizon with a linear head.

Core code lives under `src/variable_patchtst_project/`:

| Module | Role |
|--------|------|
| `utils/data_loader.py` | Electricity load dataset (KaggleHub), `ElectricityLoadDataset` pulls and creates necessary `Dataset` objects |
| `models/patching.py` | Patch boundary computation and patch algorithms |
| `models/simple_patchtst.py` | `SimplePatchTST` for model, `Patchifier` for smart patch creation, `DynamicEmbedding` for automatic embedding layer creation depending on incoming patch size, and `PositionalEncoding` |

Experiment hyperparameters and sweep rows are in **`config/`** as CSV files (`configs-32.csv`, `configs-42.csv`, `configs-64.csv`). Each CSV specifies an experiment config on a different total number of patch tokens. Each file begins with the first two row signifying a corresponding uniform patching baseline, with the rest for dynamic patching. 

## Requirements

Python 3.10+ recommended. Dependencies are listed in `pyproject.toml`: **PyTorch**, **NumPy**, **Pandas**, **kagglehub** (for downloading the electricity dataset).

Optional (used in the experiment notebook for FLOPs / memory tooling): **ptflops**.

## Data

The default experiment pipeline uses the [Electricity Load Forecasting](https://www.kaggle.com/datasets/saurabhshahane/electricity-load-forecasting) dataset via **KaggleHub**.

- **Google Colab**: the loader can mount Drive and cache data under a Drive folder (see `COLAB_DATA_PATH` in `data_loader.py`).
- **Local**: with Colab imports unavailable, downloads are placed under **`data/`** relative to the process working directory

You need Kaggle credentials configured for **kagglehub** where required (see [KaggleHub](https://github.com/Kaggle/kagglehub) documentation).

## Running experiments

The intended workflow is **`notebooks/01_Experiments.ipynb`**:

1. Install the package (and optional `ptflops` if you use those cells).
2. Load data and train/evaluate using the functions defined in the notebook.
3. Call `run_configs_and_output(run_id)` with a `run_id` that matches a config file suffix, e.g. **`32`**, **`42`**, or **`64`** for `config/configs-{run_id}.csv`. The notebook writes timestamped result CSVs under a Drive output directory when run in Colab.

For visualization of patch layouts, see **`notebooks/02_Patch_Plots.ipynb`**.

To reproduce the course experiments in **Google Colab**, simply copy or open `01_Experiments.ipynb` in Colab, run the setup cells (clone or pull this repo, install dependencies), then execute the training and sweep cells.
