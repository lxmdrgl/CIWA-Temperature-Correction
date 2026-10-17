from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from data_preprocessing import sunlit_mask, tomato_path


results_path = Path(__file__).resolve().parents[1] / "results"
input_path = results_path / "temperature_correction"
output_path = results_path / "correction_visualization"


def save_figure(fig, filename, directory=None):
    directory = output_path if directory is None else Path(directory)
    path = directory / filename
    # Windows can reject overwriting a PNG that an image viewer has memory-mapped.
    with NamedTemporaryFile(
        dir=directory, prefix=f"{path.stem}_", suffix=".png", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        fig.savefig(temporary, dpi=150)
        try:
            temporary.replace(path)
        except PermissionError:
            print(f"{path.name} is locked; saved the updated figure to {temporary}")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        plt.close(fig)


def save_distribution_versions(fig, axes, filename):
    save_figure(fig, filename)
    # Reuse the nearest panels after saving the complete figure.
    for row in (0, 2):
        for ax in axes[row]:
            ax.remove()
    grid = fig.add_gridspec(2, 3)
    fig.set_size_inches(12, 6)
    for row, original_row in enumerate((1, 3)):
        for col, ax in enumerate(axes[original_row]):
            ax.set_subplotspec(grid[row, col])
    save_figure(fig, f"{Path(filename).stem}_nearest.png")


def plot_pixel_temperatures():
    samples = {}
    for split in tqdm(("train", "valid", "test"), desc="Pixel distributions", unit="split"):
        original = tomato_path / "original" / split
        thermal_paths = sorted((original / "thermal").glob("*.npy"))
        if not thermal_paths:
            raise FileNotFoundError(original / "thermal")
        for method, interpolation in (("area", cv2.INTER_AREA), ("nearest", cv2.INTER_NEAREST)):
            groups = {}
            for source in ("original", "fcdensenet", "frrna"):
                values, weights = [], []
                for path in tqdm(
                    thermal_paths, desc=f"{split} / {method} / {source}",
                    unit="image", leave=False,
                ):
                    thermal = np.load(path)
                    mask_path = (
                        original / "label" / f"{path.stem}_L.png"
                        if source == "original" else
                        tomato_path / source / split / "prediction" / f"{path.stem}_prediction.png"
                    )
                    coverage = cv2.resize(
                        sunlit_mask(mask_path),
                        (thermal.shape[1], thermal.shape[0]),
                        interpolation=interpolation,
                    )
                    valid = np.isfinite(thermal) & (coverage > 0)
                    values.append(thermal[valid])
                    weights.append(coverage[valid])
                groups[source] = (np.concatenate(values), np.concatenate(weights))
            samples[method, split] = groups
    bins = np.histogram_bin_edges(
        np.concatenate([values for groups in samples.values() for values, _ in groups.values()]),
        bins=50,
    )
    fig, axes = plt.subplots(
        4, 3, figsize=(12, 11), sharex=True, sharey="col", layout="constrained"
    )
    for row, model in enumerate(("fcdensenet", "frrna")):
        for index, method in enumerate(("area", "nearest")):
            for col, split in enumerate(("train", "valid", "test")):
                ax = axes[row * 2 + index, col]
                means = {}
                for source, color in zip(("original", model), ("C0", "C1")):
                    values, weights = samples[method, split][source]
                    means[source] = np.average(values, weights=weights) if values.size else np.nan
                    if values.size:
                        ax.hist(
                            values, bins=bins, weights=weights, density=True,
                            histtype="step", linewidth=1.5, label=source, color=color,
                        )
                        ax.axvline(
                            means[source],
                            color=color, linestyle=":", linewidth=1.5,
                        )
                difference = means["original"] - means[model]
                ax.set_title(
                    f"{model} / {split} / {method}\n"
                    f"Original − predicted mean: {difference:+.2f} °C",
                    fontsize=10,
                )
                ax.legend()
    fig.suptitle("Sunlit pixel temperature distributions")
    fig.supxlabel("Temperature (°C)")
    fig.supylabel("Probability density (1/°C)")
    save_distribution_versions(fig, axes, "pixel_temperature_distributions.png")


def plot_area_comparison():
    methods = ("area", "area_threshold")
    temperatures = {
        (model, split, method): np.load(
            input_path / f"tomato_{model}_{split}_temperature_{method}.npy"
        )
        for model in ("fcdensenet", "frrna")
        for split in ("train", "valid", "test")
        for method in methods
    }
    all_values = np.concatenate([data.ravel() for data in temperatures.values()])
    bins = np.histogram_bin_edges(all_values[np.isfinite(all_values)], bins=50)
    fig, axes = plt.subplots(
        4, 3, figsize=(12, 11), sharex=True, sharey="col", layout="constrained"
    )
    for row, model in enumerate(("fcdensenet", "frrna")):
        for index, method in enumerate(methods):
            for col, split in enumerate(("train", "valid", "test")):
                ax = axes[row * 2 + index, col]
                means = []
                for data, label, color in zip(
                    temperatures[model, split, method].T, ("label", "predicted"), ("C0", "C1")
                ):
                    data = data[np.isfinite(data)]
                    mean = data.mean() if data.size else np.nan
                    means.append(mean)
                    ax.hist(data, bins=bins, histtype="step", linewidth=1.5, label=label, color=color)
                    if data.size:
                        ax.axvline(mean, color=color, linestyle="--", linewidth=1.5)
                description = "area weights" if method == "area" else "area threshold ≥ 50%"
                ax.set_title(
                    f"{model} / {split} / {description}\n"
                    f"Label − predicted mean: {means[0] - means[1]:+.2f} °C",
                    fontsize=10,
                )
                ax.legend()
    fig.suptitle("Per-image mean temperatures: area weights vs. thresholding")
    fig.supxlabel("Mean temperature (°C)")
    fig.supylabel("Number of images")
    save_figure(fig, "area_weighted_vs_threshold_distributions.png")


def main():
    output_path.mkdir(parents=True, exist_ok=True)
    temperature_paths = sorted(
        path
        for method in ("area", "nearest")
        for path in input_path.glob(f"tomato_*_temperature_{method}.npy")
    )
    temperature_data = {
        path.name: np.load(path)
        for path in tqdm(temperature_paths, desc="Loading temperatures", unit="file")
    }
    all_temperatures = np.concatenate([data.ravel() for data in temperature_data.values()])
    mean_bins = np.histogram_bin_edges(
        all_temperatures[np.isfinite(all_temperatures)], bins=50
    )
    all_corrections = np.concatenate([
        data[:, 0] - data[:, 1] for data in temperature_data.values()
    ])
    correction_bins = np.histogram_bin_edges(
        all_corrections[np.isfinite(all_corrections)], bins=50
    )
    fig, axes = plt.subplots(
        4, 3, figsize=(12, 11), sharex=True, sharey="col", layout="constrained"
    )
    mean_fig, mean_axes = plt.subplots(
        4, 3, figsize=(12, 11), sharex=True, sharey="col", layout="constrained"
    )
    for row, model in enumerate(("fcdensenet", "frrna")):
        for col, split in enumerate(("train", "valid", "test")):
            for index, method in enumerate(("area", "nearest")):
                temperatures = temperature_data[
                    f"tomato_{model}_{split}_temperature_{method}.npy"
                ]
                # Columns: label mean, predicted mean.
                data = temperatures[:, 0] - temperatures[:, 1]
                data = data[np.isfinite(data)]
                ax = axes[row * 2 + index, col]
                color = f"C{index}"
                mean = data.mean() if data.size else np.nan
                ax.hist(
                    data, bins=correction_bins, histtype="step", linewidth=1.5,
                    label=f"Mean: {mean:+.2f} °C", color=color,
                )
                if data.size:
                    ax.axvline(mean, color=color, linestyle="--", linewidth=1.5)
                ax.axvline(0, color="gray", linestyle="--", linewidth=1)
                ax.set_title(f"{model} / {split} / {method}")
                ax.legend()
                means = [column[np.isfinite(column)] for column in temperatures.T]
                mean_ax = mean_axes[row * 2 + index, col]
                for data, label, color in zip(means, ("label", "predicted"), ("C0", "C1")):
                    mean_ax.hist(
                        data, bins=mean_bins, histtype="step", linewidth=1.5,
                        label=label, color=color,
                    )
                    if data.size:
                        mean_ax.axvline(data.mean(), color=color, linestyle="--", linewidth=1.5)
                difference = (
                    means[0].mean() - means[1].mean()
                    if all(column.size for column in means) else np.nan
                )
                mean_ax.set_title(
                    f"{model} / {split} / {method}\n"
                    f"Label − predicted mean: {difference:+.2f} °C",
                    fontsize=10,
                )
                mean_ax.legend()
    fig.suptitle("Temperature corrections: label mean − predicted mean")
    fig.supxlabel("Correction (°C)")
    fig.supylabel("Number of images")
    save_distribution_versions(fig, axes, "correction_distributions.png")
    mean_fig.suptitle("Label and predicted mean temperatures")
    mean_fig.supxlabel("Mean temperature (°C)")
    mean_fig.supylabel("Number of images")
    save_distribution_versions(mean_fig, mean_axes, "mean_temperature_distributions.png")
    plot_pixel_temperatures()
    plot_area_comparison()


if __name__ == "__main__":
    main()
