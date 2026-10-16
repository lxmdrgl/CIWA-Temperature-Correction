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

## Returning to the project

Activate the virtual environment each time you open a new terminal:

```powershell
.\.venv\Scripts\Activate.ps1
```

To leave the virtual environment:

```powershell
deactivate
```
