"""
Use config test and split to isolate an element in the dataset

Use dataset to extract the preprocessed data and label according to the config

For thermal_data or visual_data, draw each channel image with its corresponding names
    thermal_data: temperatures, (optional: hard mask), (optional: soft mask)
    visual_data: RGB

Write the thermal_label_target temperature

Write the thermal_hard_target baseline temperature 

include the rmse, mse, mae, mbe for thermal_label_target (actual) and thermal_hard_target(prediction)

include config 

write the script as simple and short as possible
do not introduce necessarcy complexity

save in visualize folder 
"""

from pathlib import Path
from textwrap import fill

import cv2
import hydra
import numpy as np
from hydra.utils import to_absolute_path
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf

from evaluate import calculate_metrics
from utils.dataset import TemperatureDataset


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    split, index = cfg.visualize.split, cfg.visualize.index
    if split not in {"train", "valid", "test"}:
        raise ValueError("visualize.split must be train, valid, or test")
    settings = OmegaConf.to_container(cfg.dataset, resolve=True)
    settings["root"] = to_absolute_path(settings["root"])
    if settings["mask"] == "both":
        settings["mask"] = "all"
    if settings["image"] not in {"thermal", "visual"}:
        raise ValueError("dataset.image must be thermal or visual")
    if settings["mask"] not in {"all", "hard", "soft"}:
        raise ValueError("dataset.mask must be both, all, hard, or soft")
    if settings["target"] not in {"direct", "correction"}:
        raise ValueError("dataset.target must be direct or correction")
    dataset = TemperatureDataset(split=split, **settings)
    if not 0 <= index < len(dataset):
        raise ValueError(f"visualize.index must be between 0 and {len(dataset) - 1}")
    data, label = dataset[index]

    # Reconstruct absolute temperatures even when the label is a correction.
    thermal = dataset.thermal_images[index]
    interpolation = {"nearest": cv2.INTER_NEAREST, "area": cv2.INTER_AREA}[settings["interpolation"]]
    hard_mask = cv2.resize(dataset.visual_hard_masks[index].astype(np.float32),
                           (thermal.shape[1], thermal.shape[0]), interpolation=interpolation)
    baseline = float(np.sum(thermal * hard_mask) / hard_mask.sum())
    actual = float(label + baseline if settings["target"] == "correction" else label)
    metrics = calculate_metrics([actual], [baseline])

    if settings["image"] == "visual":
        # OpenCV loads BGR; display RGB without changing the dataset input.
        channels = [("RGB", cv2.cvtColor(data, cv2.COLOR_BGR2RGB), None)]
    else:
        names = ["Temperatures"]
        if settings["mask"] in {"all", "hard"}:
            names.append("Hard mask")
        if settings["mask"] in {"all", "soft"}:
            names.append("Soft mask")
        channels = [(name, data[..., i], "inferno" if i == 0 else "gray")
                    for i, name in enumerate(names)]

    lines = [f"thermal_label_target (actual): {actual:.4f} °C",
             f"thermal_hard_target (baseline): {baseline:.4f} °C",
             f"Configured label ({settings['target']}): {float(label):.4f}",
             " | ".join(
                 f"{name.upper()}: {metrics[name]:.4f}" for name in ("rmse", "mse", "mae", "mbe")),
             "MBE = baseline - actual", "", "Configuration:"]
    for section, values in OmegaConf.to_container(cfg, resolve=True).items():
        text = ", ".join(f"{key}={value}" for key, value in values.items()) if isinstance(values, dict) else str(values)
        lines.append(fill(f"{section}: {text}", width=110, subsequent_indent="  "))
    details = "\n".join(lines)
    text_height = 0.22 * len(details.splitlines())
    figure = Figure(figsize=(12, 3.5 + text_height), layout="constrained")
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(2, len(channels), height_ratios=[3.5, text_height])
    figure.suptitle(f"{cfg.dataset.crop} | {split} | index {index}")
    for i, (name, image, cmap) in enumerate(channels):
        axes = figure.add_subplot(grid[0, i])
        limits = {"vmin": 0, "vmax": 1} if "mask" in name else {}
        shown = axes.imshow(image, cmap=cmap, **limits)
        axes.set_title(name)
        axes.set_axis_off()
        if cmap:
            figure.colorbar(shown, ax=axes, shrink=0.8)
    info = figure.add_subplot(grid[1, :])
    info.set_axis_off()
    info.text(0, 1, details, transform=info.transAxes, va="top", fontsize=9, family="monospace")

    output = Path(to_absolute_path("visualize"))
    output.mkdir(parents=True, exist_ok=True)
    name = "_".join(str(cfg.dataset[key]) for key in
                    ("crop", "model", "image", "mask", "interpolation", "target"))
    path = output / f"{name}_{split}_{index}.png"
    figure.savefig(path, dpi=150)
    print(f"Saved visualization: {path}")


if __name__ == "__main__":
    main()
