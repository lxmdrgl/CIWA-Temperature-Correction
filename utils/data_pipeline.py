import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import cv2
from utils.dataset import TemperatureDataset

def load_dataset(cfg, split="train", return_metadata=False):
    """Load float32 inputs (N, C, H, W) and regression targets (N, 1)."""
    mask = cfg.dataset.mask

    if cfg.dataset.image not in {"thermal", "visual"}:
        raise ValueError(f"Invalid image type: {cfg.dataset.image}")
    if mask not in {"none", "hard", "soft", "both"}:
        raise ValueError(f"Invalid mask: {mask}")
    if cfg.dataset.target not in {"direct", "correction"}:
        raise ValueError(f"Invalid target: {cfg.dataset.target}")

    dataset = TemperatureDataset(
        root=cfg.dataset.root,
        split=split,
        crop=cfg.dataset.crop,
        model=cfg.dataset.model,
        image=cfg.dataset.image,
        mask=mask,
        interpolation=cfg.dataset.interpolation,
        target=cfg.dataset.target,
        mapping=getattr(cfg.dataset, "mapping", "original"),
        mask_fill=getattr(cfg.dataset, "mask_fill", 0.0),
    )

    images = []
    targets = []
    baselines = []
    for index in range(len(dataset)):
        image, target = dataset[index]
        images.append(image)
        targets.append(target)
        if return_metadata:
            thermal = dataset.thermal_images[index]
            interpolation = {"nearest": cv2.INTER_NEAREST, "area": cv2.INTER_AREA}[
                cfg.dataset.interpolation
            ]
            weights = cv2.resize(
                dataset.visual_hard_masks[index].astype(np.float32),
                (thermal.shape[1], thermal.shape[0]),
                interpolation=interpolation,
            )
            baselines.append(float(np.sum(thermal * weights) / weights.sum()))

    if not images:
        raise ValueError(f"No samples found for split: {split}")
    if any(image.shape != images[0].shape for image in images):
        raise ValueError(
            "Images must have the same shape before batching. "
            "Resize them to a common size before loading this split."
        )

    data = torch.from_numpy(np.stack(images)).float()
    # (N, C, H, W)
    data = data.permute(0, 3, 1, 2).contiguous()
    # (N, 1)
    labels = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)

    if return_metadata:
        baseline = torch.tensor(baselines, dtype=torch.float64).reshape(-1, 1)
        reference = torch.tensor(targets, dtype=torch.float64).reshape(-1, 1)
        if cfg.dataset.target == "correction":
            reference = reference + baseline
        return data, labels, {"baseline": baseline, "reference": reference}
    return data, labels

def create_loader(data, labels, batch_size, shuffle):
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
