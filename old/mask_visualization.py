import csv
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from data_preprocessing import sunlit_mask, tomato_path
from correction_visualization import save_figure


output_path = Path(__file__).resolve().parents[1] / "results" / "mask_visualization"
# BGR colors indexed by 2 * label + prediction: TN red, FP blue, FN yellow, TP green.
colors = np.array([(42, 42, 165), (255, 0, 0), (0, 255, 255), (0, 255, 0)], dtype=np.uint8)
interpolations = {"nearest": cv2.INTER_NEAREST, "area": cv2.INTER_AREA}


def save_overlay(image, destination):
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError(f"Could not encode {destination}")
    with NamedTemporaryFile(
        dir=destination.parent, prefix=f"{destination.stem}_", suffix=".png", delete=False
    ) as file:
        file.write(encoded.tobytes())
        temporary = Path(file.name)
    try:
        temporary.replace(destination)
    except PermissionError:
        print(f"{destination.name} is locked; saved updated overlay to {temporary}")


def downsample_mask(mask, size, interpolation):
    # Area interpolation gives fractional coverage; use majority coverage as the class.
    coverage = cv2.resize(mask.astype(np.float32), size, interpolation=interpolation)
    return (coverage >= 0.5).astype(np.uint8)


def plot_confusion_matrices():
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    # Reorder overlay colors to [[TP, FN], [FP, TN]] and convert BGR to RGB.
    cell_colors = colors[np.array([[3, 2], [1, 0]])][:, :, ::-1]
    for model_index, model in enumerate(("fcdensenet", "frrna")):
        for method_index, method in enumerate(("original", "nearest", "area")):
            ax = axes[model_index, method_index]
            suffix = "" if method == "original" else f"_{method}"
            matrix = sum(
                np.load(output_path / model / split / f"confusion_matrix{suffix}.npy")
                for split in ("train", "valid", "test")
            )
            percentages = 100 * matrix[::-1, ::-1] / matrix.sum()
            ax.imshow(cell_colors)
            for row in range(2):
                for col in range(2):
                    category = (("TP", "FN"), ("FP", "TN"))[row][col]
                    ax.text(
                        col, row, f"{category}\n{percentages[row, col]:.2f}%",
                        ha="center", va="center",
                        color="black" if row == 0 else "white",
                    )
            ax.set_title(f"{model} / {method}")
            ax.set_xticks((0, 1), ("Sunlit", "Not sunlit"))
            ax.set_yticks((0, 1), ("Sunlit", "Not sunlit"))
    fig.suptitle("Confusion matrices — all splits (each totals 100%)\nArea: sunlit coverage ≥ 50%")
    fig.supxlabel("Predicted mask")
    fig.supylabel("Original label")
    save_figure(fig, "confusion_matrices.png", directory=output_path)
    for ax in axes[:, 2]:
        ax.remove()
    grid = fig.add_gridspec(2, 2)
    fig.set_size_inches(8, 7)
    for row in range(2):
        for col in range(2):
            axes[row, col].set_subplotspec(grid[row, col])
    fig.suptitle("Confusion matrices — all splits (each totals 100%)")
    save_figure(fig, "confusion_matrices_without_area.png", directory=output_path)


def main():
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "confusion_counts.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("split", "model", "method", "image", "TN", "FP", "FN", "TP"))
        for split in ("train", "valid", "test"):
            label_paths = sorted((tomato_path / "original" / split / "label").glob("*_L.png"))
            if not label_paths:
                raise FileNotFoundError(tomato_path / "original" / split / "label")
            matrices = {
                model: {method: np.zeros((2, 2), dtype=np.int64)
                        for method in ("original", "nearest", "area")}
                for model in ("fcdensenet", "frrna")
            }
            for model in matrices:
                (output_path / model / split).mkdir(parents=True, exist_ok=True)
            for path in tqdm(label_paths, desc=f"Masks: {split}", unit="image"):
                name = path.stem.removesuffix("_L")
                label = sunlit_mask(path).astype(np.uint8)
                thermal_shape = np.load(
                    tomato_path / "original" / split / "thermal" / f"{name}.npy",
                    mmap_mode="r",
                ).shape
                size = (thermal_shape[1], thermal_shape[0])
                labels = {"original": label}
                labels.update({method: downsample_mask(label, size, interpolation)
                               for method, interpolation in interpolations.items()})
                for model, model_matrices in matrices.items():
                    prediction = sunlit_mask(
                        tomato_path / model / split / "prediction" / f"{name}_prediction.png"
                    ).astype(np.uint8)
                    if label.shape != prediction.shape:
                        raise ValueError(f"Mask shapes differ: {path.name}, {model}")
                    for method, label_mask in labels.items():
                        prediction_mask = (
                            prediction if method == "original" else
                            downsample_mask(prediction, size, interpolations[method])
                        )
                        comparison = 2 * label_mask + prediction_mask
                        counts = np.bincount(comparison.ravel(), minlength=4).reshape(2, 2)
                        model_matrices[method] += counts
                        writer.writerow((split, model, method, name, *counts.ravel()))
                        suffix = "" if method == "original" else f"_{method}"
                        destination = output_path / model / split / f"{name}_overlay{suffix}.png"
                        save_overlay(colors[comparison], destination)
            for model, model_matrices in matrices.items():
                for method, matrix in model_matrices.items():
                    # Rows: label [negative, sunlit]; columns: prediction [negative, sunlit].
                    suffix = "" if method == "original" else f"_{method}"
                    np.save(output_path / model / split / f"confusion_matrix{suffix}.npy", matrix)
    plot_confusion_matrices()


if __name__ == "__main__":
    main()
