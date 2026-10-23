"""Plot sample inputs and the thermal label mask, with temperatures and metrics."""

from pathlib import Path
from textwrap import fill

import cv2
import hydra
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
    # Always retrieve the direct label so both targets can be displayed.
    settings["target"] = "direct"
    if settings["image"] not in {"thermal", "visual"}:
        raise ValueError("dataset.image must be thermal or visual")
    if settings["mask"] not in {"none", "hard", "soft", "both"}:
        raise ValueError("dataset.mask must be none, hard, soft, or both")
    dataset = TemperatureDataset(split=split, **settings)
    if not 0 <= index < len(dataset):
        raise ValueError(f"visualize.index must be between 0 and {len(dataset) - 1}")
    data, target, metadata = dataset.get_sample(index)

    baseline = float(metadata["baseline"])
    actual = float(target)
    correction = actual - baseline
    metrics = calculate_metrics([actual], [baseline])

    if settings["image"] == "visual":
        # OpenCV loads BGR; display RGB without changing the dataset input.
        channels = [("RGB", cv2.cvtColor(data, cv2.COLOR_BGR2RGB), None)]
    else:
        names = ["Temperatures"]
        if settings["mask"] in {"both", "hard"}:
            names.append("Hard mask")
        if settings["mask"] in {"both", "soft"}:
            names.append("Soft mask")
        mapped = settings.get("mapping", "original") == "thermal"
        channels = [(f"{name} × temperature" if mapped and i > 0 else name,
                     data[..., i], "inferno" if i == 0 or mapped else "gray")
                    for i, name in enumerate(names)]
    channels.append(("label", metadata["thermal_label_mask"], "gray"))

    lines = [f"thermal_label_target (actual): {actual:.4f} °C",
             f"thermal_hard_target (baseline): {baseline:.4f} °C",
             f"Label (direct): {actual:.4f} °C",
             f"Label (correction): {correction:.4f} °C",
             " | ".join(
                 f"{name.upper()}: {metrics[name]:.4f}" for name in ("rmse", "mse", "mae", "mbe")),
             "", "Dataset configuration:"]
    text = ", ".join(f"{key}={value}" for key, value in
                     OmegaConf.to_container(cfg.dataset, resolve=True).items() if key != "target")
    lines.append(fill(text, width=110, subsequent_indent="  "))
    details = "\n".join(lines)
    text_height = 0.22 * len(details.splitlines())
    figure = Figure(figsize=(max(12, 4 * len(channels)), 3.5 + text_height), layout="constrained")
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(2, len(channels), height_ratios=[3.5, text_height])
    figure.suptitle(f"{cfg.dataset.crop} | {cfg.dataset.model} | {split} | index {index}")
    for i, (name, image, cmap) in enumerate(channels):
        axes = figure.add_subplot(grid[0, i])
        binary_or_soft = "mask" in name and "× temperature" not in name
        limits = {"vmin": 0, "vmax": 1} if binary_or_soft or name == "label" else {}
        if name == "label":
            limits["interpolation"] = "nearest"
        shown = axes.imshow(image, cmap=cmap, **limits)
        axes.set_title(name)
        axes.set_axis_off()
        if cmap:
            figure.colorbar(shown, ax=axes, shrink=0.8)
    info = figure.add_subplot(grid[1, :])
    info.set_axis_off()
    info.text(0, 1, details, transform=info.transAxes, va="top", fontsize=9, family="monospace")

    output = Path(to_absolute_path(cfg.visualize.output)) / str(index)
    output.mkdir(parents=True, exist_ok=True)
    name = "_".join(str(cfg.dataset[key]) for key in
                    ("crop", "model", "image", "mask", "interpolation"))
    if settings["image"] == "thermal" and settings.get("mapping", "original") == "thermal":
        name += f"_mapping-thermal_fill{settings.get('mask_fill', 0.0)}"
    path = output / f"{name}_{split}_{index}.png"
    figure.savefig(path, dpi=150)
    print(f"Saved visualization: {path}")


if __name__ == "__main__":
    main()
