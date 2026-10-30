# CIWA-Temperature-Regression

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
torchvision's corresponding ImageNet weights and a new regression
layer. They share the `transfer` settings. The first pretrained run downloads
the selected model's weights;
later runs use the local cache. `transfer.pretrained=false` trains from
scratch. Both direct and correction targets are supported.

All models use `dataset.reg_output=canopy` (default) for one canopy-mean
temperature output, or `dataset.reg_output=cwsi` for three temperature outputs
in this order: canopy mean, upper-tail mean, lower-tail mean. These are the
three temperatures used for CWSI, rather than a calculated CWSI index.
`dataset.bound` sets the percentage of canopy pixels in each tail (default 5%).
For correction targets, each output learns the difference from its corresponding
hard-mask baseline. Training loss averages over samples and outputs.

```powershell
python main.py dataset.reg_output=cwsi dataset.bound=5.0
```

Evaluation restores the output mode from the checkpoint and writes a separate
metrics row for each temperature, identified by `output` (`canopy`, `upper_tail`,
`lower_tail`). `samples` counts images for each row. CSV rows also include
`reg_output` and `bound` in every column mode; use a new results file if an
existing CSV has the older columns. Older checkpoints without `reg_output`
continue to use one canopy output. Visualizations display the corresponding
direct and correction temperatures for each output. Checkpoint and visualization
filenames include the output mode and, for CWSI, the tail percentage.

`evaluate.metric=all` (default) reports each temperature separately.
`evaluate.metric=avg` reports one row with `output=avg`, calculating metrics
over all temperature values from all images together (3 × image count for CWSI).
It does not average the three reported metric scores or average temperatures
before comparing them. `samples` still counts images. The setting applies to
both model and baseline evaluation; canopy mode combines its single output.

For thermal images, the first convolution supports temperature plus the selected
mask channels. Pretrained RGB filters are summed into the temperature channel;
`transfer.mask_init=zero` starts mask weights at zero (the default), while
`transfer.mask_init=default` keeps PyTorch's default random initialization
for those weights. This option also applies when training from scratch; it has no
effect on temperature-only or visual inputs. Mask weights are learned when
`transfer.freeze_conv1=false`.

`dataset.mask` selects input channels: `none` uses temperature only; `hard` adds
the hard mask; `soft` adds the probability mask; `both` adds hard and soft masks,
in that order. The hard mask is still used to calculate the correction target and
baseline even when it is excluded from the inputs.

`dataset.image=visual` converts OpenCV's BGR image to RGB and preserves its
original spatial resolution.

`dataset.image=both` resizes the visual image to the thermal image's width and
height with area interpolation and prepends its channels in RGB order. Inputs
are `[R, G, B, T]`, `[R, G, B, T, hard]`, `[R, G, B, T, soft]`, or
`[R, G, B, T, hard, soft]` according to `dataset.mask`. The loader returns
float32 tensors in PyTorch's `(N, C, H, W)` layout. The loader retains RGB on
the 0–255 scale; the model divides color channels by 255 before further
normalization. Hard and soft masks retain their original values.

For combined inputs, `train.normalize=temp` standardizes T and leaves masks
unchanged; RGB uses the model-specific normalization below. Combined ResNet
inputs initialize temperature from the mean of the pretrained RGB filters, then
multiply all four RGB/temperature filters by `3/4`. Their sum equals the original
RGB filter sum, regardless of the number of optional mask channels. Thermal-only
inputs retain the summed RGB temperature filter. Masks follow `transfer.mask_init`.

For `image=visual` or `image=both`, all models first divide RGB by 255, then
standardize each color channel. MLP and CNN use means and standard deviations
fitted across all training images and spatial pixels, separately for R, G, and B.
ResNet-18/34/50 use fixed ImageNet means `[0.485, 0.456, 0.406]` and standard
deviations `[0.229, 0.224, 0.225]`, including when training from scratch.
RGB normalization is applied for every `train.normalize` mode.

`train.normalize` controls temperature and mask channels:

- `none`: leave thermal and mask channels unchanged.
- `temp` (default): standardize the temperature channel;
  masks retain their existing 0–1 values.
- `all`: standardize temperature and each mask channel independently. RGB keeps
  its model-specific normalization and is not standardized a second time.

Statistics are calculated across training samples and spatial pixels only, then
reused for validation, test, and inference. Standard deviations are clamped to
at least `1e-6` to handle constant channels. Target labels, baseline temperatures,
and reference temperatures retain their original units. Existing BatchNorm layers
remain in place.

Normalization statistics are saved inside the model checkpoint and restored
automatically during evaluation. The saved `train.normalize` setting determines
how thermal and mask channels are normalized. MLP/CNN RGB statistics are also
saved in checkpoints and reused for validation, test, and inference.

For `image=visual` or `image=both`, color channels are divided by 255 in
`models/normalization.py` during every forward pass, including evaluation.
Standardization follows scaling, so final color values can fall outside 0–1.
No center crop is applied;
combined inputs use the thermal image resolution.

By default, all ResNet layers are fine-tuned. Set
`transfer.freeze_backbone=true` to freeze residual blocks and BatchNorm
parameters/statistics while keeping the regression head trainable.
`transfer.freeze_conv1` independently controls the first convolution:
`false` trains it, and `true` freezes it, regardless of `freeze_backbone`.
Trainable first-convolution weights use `transfer.lr_conv1`;
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
`interpolation` and `target`; `train` adds `normalize`, `mask_init`, `conv1`, and `fc`;
`all` includes both groups. `epoch`, `model`, `crop`, `seg model`, and `image`
are always included, along with metrics, seed, split, sample count, kind, saved
config, and checkpoint path. `conv1` is its saved learning rate, or zero when
frozen (also zero for MLP/CNN); `fc` is the saved `train.learning_rate`.
Checkpoint names for `train.name=train` and `train.name=all` include
`maskinit-zero` or `maskinit-default` from `transfer.mask_init`.
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
