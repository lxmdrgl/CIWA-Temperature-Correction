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


def evaluate_checkpoint(checkpoint_path, split="test", config_path=None,
                        overrides=None, output="results/metrics.csv", include_baseline=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    saved_config = checkpoint.get("config")
    if saved_config is not None:
        cfg = OmegaConf.create(saved_config)
    elif config_path is not None:
        cfg = OmegaConf.load(config_path)
        print("Legacy checkpoint: ensure the supplied config matches its training run.")
    else:
        raise ValueError("Legacy checkpoint has no saved config. Set evaluate.legacy_config.")
    if overrides:
        # Primarily useful to relocate the data root or configure legacy checkpoints.
        if saved_config is not None:
            allowed = {"dataset.root", "train.batch_size"}
            if any(item.split("=", 1)[0] not in allowed for item in overrides):
                raise ValueError(
                    "Saved training settings must stay fixed during evaluation. "
                    "Only dataset.root and train.batch_size may be overridden."
                )
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    cfg.dataset.root = to_absolute_path(cfg.dataset.root)
    data, labels, metadata = load_dataset(cfg, split, return_metadata=True)
    loader = create_loader(data, labels, cfg.train.batch_size, shuffle=False)
    state = checkpoint["model_state_dict"]
    # Older MLP/CNN models have no normalization buffers and expect raw inputs.
    normalized = "thermal_normalization_fitted" in state
    if cfg.train.model == "mlp":
        model = MLPRegressor(input_size=data[0].numel(), image=cfg.dataset.image,
                             normalize=normalized)
    elif cfg.train.model == "cnn":
        model = CNNRegressor(in_channels=data.shape[1], image=cfg.dataset.image,
                             normalize=normalized)
    elif cfg.train.model == "resnet18":
        from models.resnet18 import ResNet18Regressor

        # Normalization and trained weights are restored from the checkpoint.
        model = ResNet18Regressor(
            in_channels=data.shape[1], image=cfg.dataset.image, pretrained=False,
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
    prediction = torch.cat(predictions).numpy().reshape(-1)
    baseline = metadata["baseline"].numpy().reshape(-1)
    reference = metadata["reference"].numpy().reshape(-1)
    if cfg.dataset.target == "correction":
        prediction = baseline + prediction

    values = [("model", prediction)]
    if include_baseline:
        values.append(("baseline", baseline))
    return write_metrics(
        cfg, split, reference, values,
        output, checkpoint_path=str(checkpoint_path), epoch=checkpoint["epoch"],
        seed=cfg.train.get("seed", "unknown") if saved_config is not None else "unknown",
    )


def write_metrics(cfg, split, reference, predictions, output,
                  checkpoint_path="", epoch=0, seed="unknown"):
    common = {
        "epoch": epoch,
        "model": cfg.train.model,
        "crop": cfg.dataset.crop,
        "seg model": cfg.dataset.model,
        "image": cfg.dataset.image,
        "mask": cfg.dataset.mask,
        "interpolation": cfg.dataset.interpolation,
        "target": cfg.dataset.target,
        "seed": seed,
        "split": split,
        "samples": len(reference),
    }
    rows = []
    for kind, values in predictions:
        metrics = calculate_metrics(reference, values)
        rows.append({
            **common, "kind": kind, **metrics,
            "config": json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True),
            "checkpoint": checkpoint_path,
        })
        print(f"{kind} ({split}, n={len(reference)}): " + ", ".join(
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
    legacy_config = settings.get("legacy_config")
    if legacy_config is not None:
        legacy_config = to_absolute_path(legacy_config)
    rows = []
    for checkpoint in checkpoints:
        print(f"Evaluating {checkpoint}", flush=True)
        rows.extend(evaluate_checkpoint(
            checkpoint, split=settings.split, config_path=legacy_config,
            output=output, include_baseline=settings.baseline,
        ))
    return rows


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
