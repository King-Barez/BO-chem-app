# Mechanochemistry of Sulfur Ylides: ultra-fast and practical Corey-Chaykovsky reaction enabled by Bayesian Optimization

Gradio-based applications for Bayesian Optimization of chemical experiments. The repo contains:
- A single-objective app (yield vs PMI scalarization) in BO-chem-modular
- A multi-objective app (true Pareto optimization) in BO-chem-modular-2objs

Both apps provide:
- Interactive Gradio UI to view/edit datasets, generate suggestions, save results, and visualize analytics
- Bayesian Optimization via BoTorch/GPyTorch with fallback to Latin Hypercube Sampling (LHS)
- Automatic handling of categorical and continuous variables per configuration

## Features
- Gradio UI:
  - Data table with refresh, upload, and download
  - Suggest next experiments  with UI-friendly rounding
  - Single save per experiment or bulk save
  - Analysis: progression, correlations, feature importance, distributions, categorical breakdowns, scatter, suggestion frequency, Pareto
- Optimization:
  - Single-objective: GP on a distance-to-target scalarized objective (yield↑, PMI↓), qLogNEI acquisition
  - Multi-objective: independent GPs, qLogNEHVI acquisition, configurable REF_POINT for hypervolume
  - Fallback to Latin Hypercube Sampling when MIN_SAMPLES_FOR_BO not met or model fails
- Data handling:
  - Configurable schema via config.py
  - PMI auto-calculation in UI when yield is entered
  - Robust numeric coercion and missing-value handling
- GPU support (optional, if PyTorch CUDA is available)

## Tech stack
- UI: gradio
- Optimization: botorch, gpytorch, torch
- Numerics: numpy, scipy, pandas
- Visualization: matplotlib, seaborn (used in plotting utilities)

## Installation
Prerequisites:
- Python 3.10+ recommended
- pip >= 22
- For GPU: NVIDIA drivers + CUDA-compatible PyTorch (optional)

1) Clone and create a virtual environment

2) Install PyTorch (CPU or GPU)
- follow https://pytorch.org/get-started/locally/ to pick the right command 

3) Install project dependencies
- pip install -r requirements.txt

To verify:
- python -c "import torch, botorch, gpytorch, gradio, pandas; print('OK')"

## Quick start
Single-objective app:
- cd BO-chem-modular
- python app.py
- Open http://localhost:7862 (configurable from app.py)

Multi-objective app:
- cd BO-chem-modular-2objs
- python app.py
- Open http://localhost:7861 (configurable from app.py)
 
## How to use
- View Data: Inspect current dataset; click Refresh Data to reload from disk.
- Suggest & Save:
  - Choose the number of experiments to suggest and click “Suggest”
  - Fill in or adjust values (UI rounds: eta/base_eq to 2 decimals, freq to 0.5 increments, time to integer)
  - Enter yield to auto-compute PMI; save single rows or (multi-obj app) bulk-save all visible rows
- Analysis: Click “Load/Refresh” to see progression, correlations, feature importance, distributions, categorical analysis, scatter, Pareto (multi-obj), and suggestion frequency
- Dataset: Upload CSV to replace the dataset or download the current dataset

## Configuration
Edit the corresponding app’s config.py.

Common keys:
- variables_spec: list of variable dicts with name, type (continuous|categorical|objective), bounds/levels, description, maximize (for objectives)
- ORDERED_INPUT_COLS_FOR_UI: ordered input variable names used in UI and optimizer
- ORDERED_OBJECTIVE_COLS_FOR_UI: ordered objective names shown in UI
- columns: full CSV column order
- DATA_FILE_PATH: CSV path (default under data/)
- MIN_SAMPLES_FOR_BO: minimum rows required before switching from LHS to BO

Single-objective specifics (BO-chem-modular):
- The optimizer builds a scalar target based on normalized distance to yield target (1.0) and PMI target (0.0); acquisition: qLogNEI

Multi-objective specifics (BO-chem-modular-2objs):
- REF_POINT: torch.Tensor-like list defining the reference point for hypervolume in objective space
- Acquisition: qLogNEHVI; independent GPs per objective (ModelListGP)

Categoricals:
- One-hot encoded internally; ensure cat_var_details maps indices <-> levels

## Optimization details
- Models: SingleTaskGP with Standardize outcome transform
- Single-objective: qLogNoisyExpectedImprovement over normalized [0,1] input domain
- Multi-objective: qLogNoisyExpectedHypervolumeImprovement with user-provided REF_POINT
- Optimization uses optimize_acqf with multiple restarts and raw samples
- Fallback: LatinHypercube if data is insufficient (< MIN_SAMPLES_FOR_BO) or if modeling errors occur





