# Compare uncorrected temperatures and saved MLP, KNN, SVM corrections.
# MBE = mean(predicted correction - actual correction); positive means overestimation.

import csv
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plant = "tomato"
model = "fcdensenet"
root = Path(__file__).resolve().parents[1]
data_path = root / "results" / plant / model
if not data_path.is_dir():
    data_path = root / "results" / "feature_extraction" / plant / model
model_path = root / "results" / "training" / plant / model
output_path = root / "results" / "evaluation" / plant / model


def metrics(actual, predicted):
    error = predicted - actual
    mse = np.mean(error**2)
    return mse, np.sqrt(mse), np.mean(np.abs(error)), np.mean(error)


def plot_corrections(corrections):
    values = np.concatenate([values for actual, predictions in corrections.values()
                             for values in (actual, *predictions.values())])
    padding = 0.05 * (np.ptp(values) or 1)
    limits = (values.min() - padding, values.max() + padding)
    bins = np.linspace(*limits, 41)  # Same bins and X limits for every split/model.
    for kind in ("scatter", "histogram"):
        fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True, sharey=True,
                                 layout="constrained")
        for col, (split, (actual, predictions)) in enumerate(corrections.items()):
            for row, (name, predicted) in enumerate(predictions.items()):
                ax = axes[row, col]
                if kind == "scatter":
                    ax.scatter(actual, predicted, s=14, alpha=0.6, edgecolors="none")
                    ax.plot(limits, limits, "--", color="gray", linewidth=1)
                    ax.set_ylim(limits)
                    ax.set_aspect("equal", adjustable="box")
                else:
                    for data, label, color in ((actual, "Actual", "C0"),
                                               (predicted, "Predicted", "C1")):
                        ax.hist(data, bins=bins, density=True, histtype="step",
                                linewidth=1.5, color=color,
                                label=f"{label} (mean {data.mean():+.2f} °C)")
                        ax.axvline(data.mean(), color=color, linestyle="--", linewidth=1.5)
                    ax.legend(fontsize=8)
                    ax.set_ylim(bottom=0)
                ax.set_xlim(limits)
                ax.set_title(f"{name} / {split} (N={actual.size})")
                ax.grid(alpha=0.2)
        fig.suptitle(f"{plant} / {model}: correction temperatures")
        fig.supxlabel("Actual correction (°C)" if kind == "scatter" else "Correction (°C)")
        fig.supylabel("Predicted correction (°C)" if kind == "scatter"
                      else "Probability density (1/°C)")
        fig.savefig(output_path / f"correction_{kind}.png", dpi=180)
        plt.close(fig)


def main():
    models = {name: joblib.load(model_path / f"{name.lower()}.joblib")
              for name in ("MLP", "KNN", "SVM")}
    rows, corrections = [], {}
    print("Split  Model       N      RMSE       MAE       MBE")
    for split in ("train", "valid", "test"):
        with np.load(data_path / f"{split}.npz") as data:
            features = data["features"].T
            label = data["label_mean_temperature"].astype(np.float64)
            baseline = data["baseline_mean_temperature"].astype(np.float64)
        # All models and the baseline are evaluated on exactly the same images.
        valid = np.isfinite(label) & np.isfinite(baseline)
        if not valid.any():
            raise ValueError(f"{split}: no finite label/baseline temperatures")
        if not valid.all():
            print(f"{split}: skipping {(~valid).sum()} images with missing temperatures")
        features, label, baseline = features[valid], label[valid], baseline[valid]
        actual = label - baseline
        predictions = {name: pipeline.predict(features) for name, pipeline in models.items()}
        corrections[split] = (actual, predictions)
        # The uncorrected baseline predicts zero correction for every image.
        for name, predicted in {"baseline": np.zeros_like(actual), **predictions}.items():
            if not np.isfinite(predicted).all():
                raise ValueError(f"{split} / {name}: nonfinite predictions")
            scores = metrics(actual, predicted)
            rows.append((split, name, label.size, *scores))
            print(f"{split:6s} {name:8s} {label.size:4d}" +
                  "".join(f"{value:10.4f}" for value in scores[1:]))
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("split", "model", "n", "mse", "rmse", "mae",
                         "mbe"))
        writer.writerows(rows)
    plot_corrections(corrections)


if __name__ == "__main__":
    main()
