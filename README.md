# CIWA-Temperature-Correction

## Installation

Requires **Python 3.12.10**. Run the following commands in PowerShell from the project folder.

### 1. Install Python

Download and install Python 3.12.10 if it is not already installed.

### 2. Create a virtual environment

```powershell
py -V:3.12 -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

Verify that the environment uses Python 3.12.10:

```powershell
python --version
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Data folder structure

The tomato dataset is stored in `data/tomato/`. Each branch is divided into `train`, `valid` (validation), and `test` splits.

```text
data/tomato/
├── original/
│   └── {train,valid,test}/
│       ├── label/
│       ├── thermal/
│       └── visual/
└── {fcdensenet,frrna}/
    └── {train,valid,test}/
        ├── logit/
        └── prediction/
```

`original/` contains the source data and labels. `fcdensenet/` and `frrna/` contain the corresponding models' logits and predictions. The `{train,valid,test}` notation represents three separate folders.

## Evaluate model configurations

Train runs using the same splits and several seeds, for example:

```powershell
python main.py train.model=mlp train.seed=0
python main.py train.model=cnn train.seed=0
```

New checkpoints include their training configuration and seed. Evaluate each
checkpoint on validation data when comparing configurations:

```powershell
python evaluate.py --checkpoint checkpoints/YOUR_MODEL.pt
```

After selecting a configuration, use `--split test` for its final assessment.
Metrics append to `results/metrics.csv` (or the path supplied with `--output`).
Each evaluation writes model and uncorrected hard-mask baseline rows containing
R², RMSE, MSE, MAE, and MBE, together with configuration, seed, and sample count.
Correction outputs are added to the hard-mask baseline before calculating metrics
against reference temperatures. Positive MBE means overestimation. R² is `nan`
when the reference temperature is constant. Repeated evaluations append new rows.

Older checkpoints without saved configuration require
`--config configs/config.yaml`; that file must match the original training run.
Use `--override train.model=cnn` if needed for an older CNN checkpoint. For new
checkpoints only `dataset.root` and `train.batch_size` may be overridden.

## Returning to the project

Activate the virtual environment each time you open a new terminal:

```powershell
.\.venv\Scripts\Activate.ps1
```

To leave the virtual environment:

```powershell
deactivate
```
