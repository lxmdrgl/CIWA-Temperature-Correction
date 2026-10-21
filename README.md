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

For this project's NVIDIA GPU setup (CUDA 12.6), replace a CPU-only PyTorch
installation with the CUDA build after installing the requirements:

```powershell
python -m pip install torch==2.14.1+cu126 torchvision==0.29.1+cu126 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Run these commands and `python main.py` with the project's `.venv` activated.

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
python main.py train.model=resnet18 train.learning_rate=0.0001
```

ResNet-18 starts with torchvision's ImageNet weights and a new single-output
regression layer. The first pretrained run downloads the weights (about 45 MB);
later runs use the local cache. `train.resnet18.pretrained=false` trains from
scratch. Both direct and correction targets are supported.

For thermal images, the first convolution supports temperature plus the selected
mask channels. Pretrained RGB filters are summed into the temperature channel;
mask weights start at zero and are learned during training.

All three models (MLP, CNN, and ResNet-18) normalize thermal inputs with one
temperature mean and standard deviation calculated from training pixels only.
The same statistics are reused for validation and test images, preserving absolute
temperature differences. Masks keep their 0–1 values. Target labels, baseline
temperatures, and reference temperatures retain their original units. Existing
BatchNorm layers remain in place.

Normalization statistics are saved inside the model checkpoint and restored
automatically during evaluation. Older MLP/CNN checkpoints without these statistics
continue to receive raw inputs. Existing ResNet-18 checkpoints retain their saved
normalization. Retrain MLP/CNN models to use the new normalization. Visual input
preprocessing for MLP/CNN is unchanged.

For visual inputs, ResNet converts the loader's BGR images to RGB, scales by 255,
and applies ImageNet normalization. Both input types keep the full image at the
dataset's existing resolution; no center crop is applied.

By default, all ResNet layers are fine-tuned. Set
`train.resnet18.freeze_backbone=true` to train only the regression head and, for
thermal inputs, the adapted input convolution. Frozen BatchNorm statistics remain
fixed. Freezing requires pretrained weights. Evaluation creates the architecture
without downloading ImageNet weights, then restores the saved checkpoint.

To evaluate trained checkpoints, edit the `evaluate` section in
`configs/config.yaml`, then run `python evaluate.py` with no arguments:

```yaml
evaluate:
  root: checkpoints
  split: valid
  checkpoint: all
  name: null
  baseline: true
  output: results/metrics.csv
  legacy_config: null
```

`checkpoint: all` evaluates every `.pt` file directly inside `root`, sorted by
filename. To evaluate one file, set `checkpoint: single` and
`name: YOUR_MODEL.pt`. `name` is ignored in all mode. Paths are relative to the
project root unless absolute.

`baseline: true` includes an uncorrected hard-mask baseline row for each
checkpoint, using that checkpoint's dataset settings. Set it to `false` to write
only model rows. Baseline rows may repeat when checkpoints share dataset settings.
No checkpoints, a missing filename in single mode, or a missing file produces an error.

New checkpoints supply their own training configuration and seed; the top-level
`train` and `dataset` settings do not replace those saved settings. Compare
configurations on `split: valid`, then use `split: test` for the final assessment.
Metrics append to `evaluate.output`. Each row contains
R², RMSE, MSE, MAE, and MBE, together with configuration, seed, and sample count.
Correction outputs are added to the hard-mask baseline before calculating metrics
against reference temperatures. Positive MBE means overestimation. R² is `nan`
when the reference temperature is constant. Repeated evaluations append new rows.

Older checkpoints without saved configuration require `evaluate.legacy_config`
to point to a config matching their original training run. This fallback is ignored
for checkpoints that contain their own config. When evaluating all files, every
legacy checkpoint must match that fallback; otherwise evaluate them individually
with the appropriate config.

## Returning to the project

Activate the virtual environment each time you open a new terminal:

```powershell
.\.venv\Scripts\Activate.ps1
```

To leave the virtual environment:

```powershell
deactivate
```
