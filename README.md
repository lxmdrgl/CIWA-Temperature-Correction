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
python main.py train.model=resnet34 train.learning_rate=0.0001
python main.py train.model=resnet50 train.learning_rate=0.0001
```

ResNet-18, ResNet-34, and ResNet-50 are separate model choices, each using
torchvision's corresponding ImageNet weights and a new single-output regression
layer. They share the `train.transfer` settings. The first pretrained run downloads
the selected model's weights;
later runs use the local cache. `train.transfer.pretrained=false` trains from
scratch. Both direct and correction targets are supported.

For thermal images, the first convolution supports temperature plus the selected
mask channels. Pretrained RGB filters are summed into the temperature channel;
mask weights start at zero and are learned during training.

`dataset.mask` selects input channels: `none` uses temperature only; `hard` adds
the hard mask; `soft` adds the probability mask; `both` adds hard and soft masks,
in that order. The hard mask is still used to calculate the correction target and
baseline even when it is excluded from the inputs.

All models (MLP, CNN, and the three ResNet variants) support three input
normalization methods through `train.normalize`:

- `none`: pass all input channels through unchanged.
- `temp` (default): standardize only the temperature channel; masks retain their
  existing 0–1 values. For visual inputs, this mode leaves inputs unchanged.
- `all`: standardize every input channel independently, including hard and soft
  masks, using a separate mean and standard deviation for each channel.

Statistics are calculated across training samples and spatial pixels only, then
reused for validation, test, and inference. Standard deviations are clamped to
at least `1e-6` to handle constant channels. Target labels, baseline temperatures,
and reference temperatures retain their original units. Existing BatchNorm layers
remain in place.

Normalization statistics are saved inside the model checkpoint and restored
automatically during evaluation. The saved `train.normalize` setting determines
whether input normalization is enabled.

RGB normalization is not applied. Inputs keep the full image at the dataset's
existing resolution; no center crop is applied.

By default, all ResNet layers are fine-tuned. Set
`train.transfer.freeze_backbone=true` to freeze residual blocks and BatchNorm
parameters/statistics while keeping the regression head trainable.
`train.transfer.freeze_conv1` independently controls the first convolution:
`false` trains it, and `true` freezes it, regardless of `freeze_backbone`.
Trainable first-convolution weights use `train.transfer.learning_rate_conv1`;
other trainable parameters use `train.learning_rate`.
Freezing the backbone requires pretrained weights. Evaluation creates the architecture
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
```

`checkpoint: all` evaluates every `.pt` file directly inside `root`, sorted by
filename. To evaluate one file, set `checkpoint: single` and
`name: YOUR_MODEL.pt`. `name` is ignored in all mode. Paths are relative to the
project root unless absolute.

`baseline: true` writes only uncorrected hard-mask baseline rows, using each
checkpoint's dataset settings, and skips model inference. Set it to `false` to
write only model rows. Baseline rows may repeat when checkpoints share dataset settings.

`evaluate.column` selects optional CSV fields: `dataset` adds `mask`,
`interpolation`, and `target`; `train` adds `normalize`, `conv1`, and `fc`;
`all` includes both groups. `epoch`, `model`, `crop`, `seg model`, and `image`
are always included, along with metrics, seed, split, sample count, kind, saved
config, and checkpoint path. `conv1` is its saved learning rate, or zero when
frozen (also zero for MLP/CNN); `fc` is the saved `train.learning_rate`.
No checkpoints, a missing filename in single mode, or a missing file produces an error.

Checkpoints supply their own training configuration and seed; the top-level
`train` and `dataset` settings do not replace those saved settings. Compare
configurations on `split: valid`, then use `split: test` for the final assessment.
Metrics append to `evaluate.output`. Each row contains
R², RMSE, MSE, MAE, and MBE, together with configuration, seed, and sample count.
Correction outputs are added to the hard-mask baseline before calculating metrics
against reference temperatures. Positive MBE means overestimation. R² is `nan`
when the reference temperature is constant. Repeated evaluations append new rows.

Evaluation requires checkpoints with their saved training configuration, including
`train.normalize`, and matching model weights and normalization buffers.

## Returning to the project

Activate the virtual environment each time you open a new terminal:

```powershell
.\.venv\Scripts\Activate.ps1
```

To leave the virtual environment:

```powershell
deactivate
```
