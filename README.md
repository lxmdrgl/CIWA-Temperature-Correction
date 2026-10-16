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
│       ├── raw/
│       ├── thermal/
│       ├── thermal-data/
│       └── visual/
├── fcdensenet/
│   └── {train,valid,test}/
│       ├── logits/
│       └── predictions/
└── frrna/
    └── {train,valid,test}/
        ├── logits/
        └── predictions/
```

`original/` contains the source data and labels. `fcdensenet/` and `frrna/` contain the corresponding models' logits and predictions. The `{train,valid,test}` notation represents three separate folders.

## Returning to the project

Activate the virtual environment each time you open a new terminal:

```powershell
.\.venv\Scripts\Activate.ps1
```

To leave the virtual environment:

```powershell
deactivate
```
