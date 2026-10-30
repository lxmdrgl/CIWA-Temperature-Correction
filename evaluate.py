"""Evaluate checkpoints using the evaluate section of configs/config.yaml.

Run from the project root: python evaluate.py
"""

import csv
import json
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from models.mlp import MLPRegressor
from models.cnn import CNNRegressor
from utils.regression import output_names
from utils.data_pipeline import load_dataset, create_loader


def calculate_metrics(reference, prediction):
    """MBE is prediction minus reference; R² is NaN for constant references."""
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if reference.size == 0 or reference.shape != prediction.shape:
        raise ValueError("Predictions and references must have the same nonzero length")
    if not (np.isfinite(reference).all() and np.isfinite(prediction).all()):
        raise ValueError("Predictions and references must contain only finite values")

    error = prediction - reference
    mse = float(np.mean(error ** 2))
    variance_sum = np.sum((reference - reference.mean()) ** 2)
    r2 = float(1 - np.sum(error ** 2) / variance_sum) if variance_sum > 0 else float("nan")
    return {
        "r2": r2,
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "mae": float(np.mean(np.abs(error))),
        "mbe": float(np.mean(error)),
    }


def evaluate_checkpoint(checkpoint_path, split="test",
                        overrides=None, output="results/metrics.csv", include_baseline=True,
                        column="all", metric="all"):
    if metric not in {"all", "avg"}:
        raise ValueError("evaluate.metric must be all or avg")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    saved_config = checkpoint.get("config")
    if saved_config is None:
        raise ValueError("Checkpoint must contain its saved training configuration.")
    cfg = OmegaConf.create(saved_config)
    # Older checkpoints saved transfer settings inside the train section.
    if "transfer" not in cfg and "transfer" in cfg.train:
        cfg.transfer = cfg.train.pop("transfer")
    if overrides:
        allowed = {"dataset.root", "train.batch_size"}
        if any(item.split("=", 1)[0] not in allowed for item in overrides):
            raise ValueError(
                "Saved training settings must stay fixed during evaluation. "
                "Only dataset.root and train.batch_size may be overridden."
            )
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    num_outputs = len(output_names(cfg.dataset.get("reg_output", "canopy")))
    cfg.dataset.root = to_absolute_path(cfg.dataset.root)
    data, labels, metadata = load_dataset(cfg, split, return_metadata=True)
    baseline = metadata["baseline"].numpy()
    reference = metadata["reference"].numpy()
    if include_baseline:
        return write_metrics(
            cfg, split, reference, [("baseline", baseline)], output,
            checkpoint_path=str(checkpoint_path), epoch=checkpoint["epoch"],
            seed=cfg.train.seed, column=column, metric=metric,
        )
    loader = create_loader(data, labels, cfg.train.batch_size, shuffle=False)
    state = checkpoint["model_state_dict"]
    normalized = cfg.train.normalize
    if cfg.train.model == "mlp":
        model = MLPRegressor(num_outputs=num_outputs, input_size=data[0].numel(), image=cfg.dataset.image,
                             normalize=normalized, in_channels=data.shape[1])
    elif cfg.train.model == "cnn":
        model = CNNRegressor(num_outputs=num_outputs, in_channels=data.shape[1], image=cfg.dataset.image,
                             normalize=normalized)
    elif cfg.train.model in {"resnet18", "resnet34", "resnet50"}:
        from models.resnet18 import ResNet18Regressor
        from models.resnet34 import ResNet34Regressor
        from models.resnet50 import ResNet50Regressor

        # Normalization and trained weights are restored from the checkpoint.
        regressors = {"resnet18": ResNet18Regressor, "resnet34": ResNet34Regressor,
                      "resnet50": ResNet50Regressor}
        model = regressors[cfg.train.model](
            **({"crop": cfg.dataset.crop} if cfg.train.model == "resnet50" else {}),
            num_outputs=num_outputs,
            in_channels=data.shape[1], image=cfg.dataset.image, pretrained=False,
            normalize=normalized,
        )
    else:
        raise ValueError(f"Invalid training model: {cfg.train.model}")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    predictions = []
    with torch.inference_mode():
        for inputs, _ in loader:
            predictions.append(model(inputs.to(device)).cpu().double())
    prediction = torch.cat(predictions).numpy()
    if prediction.shape != reference.shape:
        raise ValueError("Model output shape must match reference shape")
    if cfg.dataset.target == "residual":
        prediction = baseline + prediction

    return write_metrics(
        cfg, split, reference, [("model", prediction)],
        output, checkpoint_path=str(checkpoint_path), epoch=checkpoint["epoch"],
        seed=cfg.train.seed, column=column, metric=metric,
    )


def write_metrics(cfg, split, reference, predictions, output,
                  checkpoint_path="", epoch=0, seed="unknown", column="all", metric="all"):
    if column not in {"dataset", "train", "all"}:
        raise ValueError("evaluate.column must be dataset, train, or all")
    if metric not in {"all", "avg"}:
        raise ValueError("evaluate.metric must be all or avg")
    common = {
        "epoch": epoch,
        "model": cfg.train.model,
        "crop": cfg.dataset.crop,
        "seg model": cfg.dataset.model,
        "image": cfg.dataset.image,
    }
    if column in {"dataset", "all"}:
        common.update(mask=cfg.dataset.mask, interpolation=cfg.dataset.interpolation,
                      target=cfg.dataset.target)
    if column in {"train", "all"}:
        common.update(
            normalize=cfg.train.normalize,
            mask_init=OmegaConf.select(cfg, "transfer.mask_init", default="zero"),
            conv1=(cfg.transfer.lr_conv1
                   if cfg.train.model in {"resnet18", "resnet34", "resnet50"}
                   and not cfg.transfer.freeze_conv1 else 0),
            fc=cfg.train.learning_rate,
        )
    names = output_names(cfg.dataset.get("reg_output", "canopy"))
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim == 1 and len(names) == 1:
        reference = reference[:, None]
    if reference.ndim != 2 or reference.shape[1] != len(names):
        raise ValueError("Reference shape must match reg_output: (N, 1) or (N, 3)")
    common.update(reg_output=cfg.dataset.get("reg_output", "canopy"),
                  bound=cfg.dataset.get("bound", 5.0),
                  seed=seed, split=split, samples=len(reference))
    rows = []
    for kind, values in predictions:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1 and len(names) == 1:
            values = values[:, None]
        if values.shape != reference.shape:
            raise ValueError("Prediction shape must match reference shape")
        groups = ([("avg", reference, values)] if metric == "avg" else
                  [(name, reference[:, index], values[:, index])
                   for index, name in enumerate(names)])
        for name, group_reference, group_prediction in groups:
            metrics = calculate_metrics(group_reference, group_prediction)
            rows.append({
                **common, "kind": kind, "output": name, **metrics,
                "config": json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True),
                "checkpoint": checkpoint_path,
            })
            print(f"{kind}/{name} ({split}, n={len(reference)}): " + ", ".join(
                f"{key}={value:.6f}" for key, value in metrics.items()
            ))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    has_header = output.exists() and output.stat().st_size > 0
    if has_header:
        with output.open(newline="", encoding="utf-8") as file:
            if next(csv.reader(file)) != list(rows[0]):
                raise ValueError("Existing results CSV has different columns; choose another evaluate.output")
    with output.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        if not has_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Results appended to {output}")
    return rows


def run_evaluation(cfg):
    settings = cfg.evaluate
    if settings.split not in {"train", "valid", "test"}:
        raise ValueError("evaluate.split must be train, valid, or test")
    if not isinstance(settings.baseline, bool):
        raise ValueError("evaluate.baseline must be true or false")
    column = settings.get("column", "all")
    if column not in {"dataset", "train", "all"}:
        raise ValueError("evaluate.column must be dataset, train, or all")
    metric = settings.get("metric", "all")
    if metric not in {"all", "avg"}:
        raise ValueError("evaluate.metric must be all or avg")

    root = Path(to_absolute_path(settings.root))
    if settings.checkpoint == "all":
        checkpoints = sorted(path for path in root.glob("*.pt") if path.is_file())
        if not checkpoints:
            raise ValueError(f"No .pt checkpoints found in {root}")
    elif settings.checkpoint == "single":
        if not settings.name:
            raise ValueError("Set evaluate.name to a checkpoint filename for single mode")
        checkpoint = root / settings.name
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        checkpoints = [checkpoint]
    else:
        raise ValueError("evaluate.checkpoint must be all or single")

    output = to_absolute_path(settings.get("output", "results/metrics.csv"))
    rows = []
    for checkpoint in checkpoints:
        print(f"Evaluating {checkpoint}", flush=True)
        rows.extend(evaluate_checkpoint(
            checkpoint, split=settings.split,
            output=output, include_baseline=settings.baseline, column=column, metric=metric,
        ))
    return rows


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
