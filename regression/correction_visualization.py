from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


results_path = Path(__file__).resolve().parents[1] / "results"
input_path = results_path / "temperature_correction"
output_path = results_path / "correction_visualization"


def main():
    output_path.mkdir(parents=True, exist_ok=True)
    temperature_data = {
        path.name: np.load(path)
        for method in ("area", "nearest")
        for path in input_path.glob(f"tomato_*_temperature_{method}.npy")
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
    fig.savefig(output_path / "correction_distributions.png", dpi=150)
    plt.close(fig)
    mean_fig.suptitle("Label and predicted mean temperatures")
    mean_fig.supxlabel("Mean temperature (°C)")
    mean_fig.supylabel("Number of images")
    mean_fig.savefig(output_path / "mean_temperature_distributions.png", dpi=150)
    plt.close(mean_fig)


if __name__ == "__main__":
    main()
