"""Evaluate a checkpoint and append temperature metrics to a comparison CSV.

Examples (from the project root):
    python evaluate.py --checkpoint checkpoints/model.pt
    python evaluate.py --checkpoint checkpoints/model.pt --split test
    python evaluate.py --checkpoint checkpoints/old.pt --config configs/config.yaml
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
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
                        overrides=None, output="results/metrics.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    saved_config = checkpoint.get("config")
    if saved_config is not None:
        cfg = OmegaConf.create(saved_config)
        if config_path is not None:
            raise ValueError("This checkpoint contains its config; omit --config")
    elif config_path is not None:
        cfg = OmegaConf.load(config_path)
        print("Legacy checkpoint: ensure the supplied config matches its training run.")
    else:
        raise ValueError("Legacy checkpoint has no saved config. Supply --config.")
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

    data, labels, metadata = load_dataset(cfg, split, return_metadata=True)
    loader = create_loader(data, labels, cfg.train.batch_size, shuffle=False)
    if cfg.train.model == "mlp":
        model = MLPRegressor(input_size=data[0].numel())
    elif cfg.train.model == "cnn":
        model = CNNRegressor(in_channels=data.shape[1])
    else:
        raise ValueError(f"Invalid training model: {cfg.train.model}")
    model.load_state_dict(checkpoint["model_state_dict"])
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

    common = {
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint["epoch"],
        "model": cfg.train.model,
        "crop": cfg.dataset.crop,
        "segmentation_model": cfg.dataset.model,
        "image": cfg.dataset.image,
        "mask": cfg.dataset.mask,
        "interpolation": cfg.dataset.interpolation,
        "target": cfg.dataset.target,
        "seed": cfg.train.get("seed", "unknown") if saved_config is not None else "unknown",
        "split": split,
        "samples": len(reference),
        "config": json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True),
    }
    rows = []
    for kind, values in (("model", prediction), ("hard_mask_baseline", baseline)):
        metrics = calculate_metrics(reference, values)
        rows.append({**common, "kind": kind, **metrics})
        print(f"{kind} ({split}, n={len(reference)}): " + ", ".join(
            f"{key}={value:.6f}" for key, value in metrics.items()
        ))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    has_header = output.exists() and output.stat().st_size > 0
    if has_header:
        with output.open(newline="", encoding="utf-8") as file:
            if next(csv.reader(file)) != list(rows[0]):
                raise ValueError("Existing results CSV has different columns; choose another --output")
    with output.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        if not has_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Results appended to {output}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--config", help="Matching config for a legacy checkpoint")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides: key=value")
    parser.add_argument("--output", default="results/metrics.csv")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.split, args.config, args.override, args.output)


if __name__ == "__main__":
    main()
