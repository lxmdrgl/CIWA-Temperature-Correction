"""Plot visual counterparts above sample inputs, with temperatures and metrics."""

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
from utils.regression import output_names


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    split, index = cfg.visualize.split, cfg.visualize.index
    if split not in {"train", "valid", "test"}:
        raise ValueError("visualize.split must be train, valid, or test")
    settings = OmegaConf.to_container(cfg.dataset, resolve=True)
    settings["root"] = to_absolute_path(settings["root"])
    # Always retrieve the direct label so both targets can be displayed.
    settings["target"] = "direct"
    if settings["image"] not in {"thermal", "visual", "both"}:
        raise ValueError("dataset.image must be thermal, visual, or both")
    if settings["mask"] not in {"none", "hard", "soft", "both"}:
        raise ValueError("dataset.mask must be none, hard, soft, or both")
    # Display every thermal channel regardless of the selected training inputs.
    display_settings = {**settings, "image": "thermal", "mask": "both"}
    dataset = TemperatureDataset(split=split, **display_settings)
    if not 0 <= index < len(dataset):
        raise ValueError(f"visualize.index must be between 0 and {len(dataset) - 1}")
    data, target, metadata = dataset.get_sample(index)

    baseline = np.asarray(metadata["baseline"]).reshape(-1)
    actual = np.asarray(target).reshape(-1)
    correction = actual - baseline

    names = ["Temperatures", "Hard mask", "Soft mask"]
    channels = [(name, data[..., i], "inferno" if i == 0 else "gray")
                for i, name in enumerate(names)]
    channels.append(("label", metadata["thermal_label_mask"], "gray"))
    visual_channels = [
        ("visual_image", cv2.cvtColor(dataset.visual_images[index], cv2.COLOR_BGR2RGB), None),
        ("visual_hard_mask", dataset.visual_hard_masks[index], "gray"),
        ("visual_soft_mask", dataset.visual_soft_masks[index], "gray"),
        ("visual_label_mask", dataset.visual_label_masks[index], "gray"),
    ]

    lines = []
    display_names = {"canopy": "Canopy", "upper_tail": "Upper bound", "lower_tail": "Lower bound"}
    for i, name in enumerate(output_names(settings.get("reg_output", "canopy"))):
        lines.append(
            f"{display_names[name]}: actual / label (direct): {actual[i]:.4f} °C | "
            f"hard-mask baseline: {baseline[i]:.4f} °C | "
            f"label (correction): {correction[i]:.4f} °C"
        )
    metrics = calculate_metrics(actual, baseline)
    cwsi = settings.get("reg_output", "canopy") == "cwsi"
    lines.append("Baseline metrics (all three temperatures):" if cwsi else "Baseline metrics (canopy):")
    metric_names = ("r2", "rmse", "mse", "mae", "mbe") if cwsi else ("rmse", "mse", "mae", "mbe")
    lines.append(" | ".join(f"{key.upper()}: {metrics[key]:.4f}" for key in metric_names))
    lines.extend(["", "Dataset configuration:"])
    text = ", ".join(f"{key}={value}" for key, value in
                     OmegaConf.to_container(cfg.dataset, resolve=True).items() if key != "target")
    lines.append(fill(text, width=110, subsequent_indent="  "))
    details = "\n".join(lines)
    text_height = 0.22 * len(details.splitlines())
    figure = Figure(figsize=(16, 7 + text_height), layout="constrained")
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(3, 4, height_ratios=[3.5, 3.5, text_height])
    figure.suptitle(f"{cfg.dataset.crop} | {cfg.dataset.model} | {split} | index {index}")
    panels = [(0, i, channel) for i, channel in enumerate(visual_channels)]
    panels.extend((1, i, channel) for i, channel in enumerate(channels))
    for row, column, (name, image, cmap) in panels:
        axes = figure.add_subplot(grid[row, column])
        binary_or_soft = "mask" in name
        limits = {"vmin": 0, "vmax": 1} if binary_or_soft or name == "label" else {}
        if name in {"label", "visual_label_mask", "visual_hard_mask"}:
            limits["interpolation"] = "nearest"
        shown = axes.imshow(image, cmap=cmap, **limits)
        axes.set_title(name)
        axes.set_axis_off()
        if cmap:
            figure.colorbar(shown, ax=axes, shrink=0.8)
    info = figure.add_subplot(grid[2, :])
    info.set_axis_off()
    info.text(0, 1, details, transform=info.transAxes, va="top", fontsize=9, family="monospace")

    output = Path(to_absolute_path(cfg.visualize.output)) / str(index)
    output.mkdir(parents=True, exist_ok=True)
    name = "_".join(str(cfg.dataset[key]) for key in
                    ("crop", "model", "image", "mask", "interpolation"))
    name += f"_{settings.get('reg_output', 'canopy')}"
    if settings.get("reg_output", "canopy") == "cwsi":
        name += f"_{settings.get('bound', 5.0)}"
    path = output / f"{name}_{split}_{index}.png"
    figure.savefig(path, dpi=150)
    print(f"Saved visualization: {path}")


if __name__ == "__main__":
    main()
