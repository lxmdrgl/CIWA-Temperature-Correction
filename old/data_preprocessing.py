from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


tomato_path = Path(__file__).resolve().parents[1] / "data" / "tomato"
output_path = Path(__file__).resolve().parents[1] / "results" / "temperature_correction"


def sunlit_mask(mask_path):
    mask = cv2.imread(str(mask_path))
    if mask is None:
        raise FileNotFoundError(mask_path)
    return np.all(mask == (0, 255, 0), axis=-1).astype(np.float32)


def mean_temperature_area(thermal, mask_path):
    # Area weights preserve partial leaf coverage when downscaling.
    weights = cv2.resize(
        sunlit_mask(mask_path),
        (thermal.shape[1], thermal.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    valid = np.isfinite(thermal) & (weights > 0)
    if not valid.any():
        return np.nan
    return np.average(thermal[valid], weights=weights[valid])


def mean_temperature_area_threshold(thermal, mask_path):
    coverage = cv2.resize(
        sunlit_mask(mask_path),
        (thermal.shape[1], thermal.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    valid = np.isfinite(thermal) & (coverage >= 0.5)
    return thermal[valid].mean() if valid.any() else np.nan


def mean_temperature_nearest(thermal, mask_path):
    mask = cv2.resize(
        sunlit_mask(mask_path),
        (thermal.shape[1], thermal.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    valid = np.isfinite(thermal) & (mask > 0)
    return thermal[valid].mean() if valid.any() else np.nan


def main():
    output_path.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        original = tomato_path / "original" / split
        thermal_paths = sorted((original / "thermal").glob("*.npy"))
        if not thermal_paths:
            raise FileNotFoundError(original / "thermal")
        for method, mean_temperature in (
            ("area", mean_temperature_area),
            ("nearest", mean_temperature_nearest),
            ("area_threshold", mean_temperature_area_threshold),
        ):
            temperatures = {model: [] for model in ("fcdensenet", "frrna")}
            for path in tqdm(thermal_paths, desc=f"{split}: {method}", unit="image"):
                thermal = np.load(path)
                label_mean = mean_temperature(
                    thermal, original / "label" / f"{path.stem}_L.png"
                )
                for model, values in temperatures.items():
                    prediction_mean = mean_temperature(
                        thermal,
                        tomato_path / model / split / "prediction"
                        / f"{path.stem}_prediction.png",
                    )
                    values.append((label_mean, prediction_mean))
            for model, values in temperatures.items():
                # Shape (images, 2): label mean, predicted mean; sorted by filename.
                # Empty sunlit masks produce NaN temperatures.
                np.save(
                    output_path / f"tomato_{model}_{split}_temperature_{method}.npy",
                    np.asarray(values, dtype=np.float32),
                )


if __name__ == "__main__":
    main()
