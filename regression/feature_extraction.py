# data/tomato/
# ├── original/
# │   └── {train,valid,test}/
# │       ├── label/
# │       ├── thermal/
# │       └── visual/
# └── {fcdensenet,frrna}/
#     └── {train,valid,test}/
#         ├── logit/
#         └── prediction/

# Regression model: predict the correction temperature
# Model: MLP in scikit learn
# correction temperature = "label" mean temperature - "prediction" mean temperature
# Loss: Mean squared error: actual correction - predicted correction
# Ground truth: "label" mask
# Input: "Logit" mask

# for each split (train,valid,test):
# only do model fcdensenet:

# preprocessing: masks and downsampling 
# - downsample "label" to thermal image using inner_nearest
# - logits (0:nonsunlit, 1:sunlit)
# - calculate 1-channel sunlit binary mask using argmax of "logit"
# - calculate 1-channel nonsunlit binary mask using complement of sunlit binary mask
# - calculate 1-channel sunlit probability mask using softmax of "logit"
# - calculate 1-channel nonsunlit probability mask 
# - downsample the four model masks to thermal image using inner_nearest
# - Result: 5 downsampled masks (1 ground truth, 4 inputs)

# Get ready for feature extraction using downsampled masks

# variables for loss:
# label mean temperature: label sunlit binary mask
# baseline mean temperature: logit sunlit binary mask

# features for model inputs:
# 5 distributions x 10 statistics = 50 features
# 5 distributions:
    # Hard sunlit leaves temperatures
        # logit mask (binary sunlit)
    # Hard nonsunlit leaves temperatures
        # logit mask (binary nonsunlit)
    # Soft weighted sunlit leaves temperature
        # logit mask (weighted probability sunlit)
    # Soft weighted nonsunlit leaves temperature
        # logit mask (weighted probability nonsunlit)
    # full image 
# 10 statistics:
    # mean
    # standard deviation
    # skewness 
    # kurtosis
    # lower quartile
    # median
    # upper quartile
    # Interquartile Range	
    # 90th percentile
    # 10th percentile

# only save 3 variables in the results per split
# (label mean temperature x num images)
# (baseline mean temperature x num images)
# (50 features x num images)

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


plant = "tomato"
model = "frrna"
root = Path(__file__).resolve().parents[1]
data_path = root / "data" / plant
output_path = root / "results" / "feature_extraction" / plant / model


def prepare_masks(label, logits, thermal_shape):
    """Resize a binary sunlit label and four logit-derived masks to thermal H x W."""
    if logits.ndim != 3 or logits.shape[0] != 2 or logits.shape[1:] != label.shape:
        raise ValueError("Expected logits (2, H, W) matching the label (H, W)")
    if not np.isfinite(logits).all():
        raise ValueError("Logits must be finite")
    binary = logits.argmax(axis=0).astype(np.float32)
    probabilities = np.exp(logits - logits.max(axis=0, keepdims=True))
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    masks = dict(label=label, sunlit_binary=binary, nonsunlit_binary=1 - binary,
                 sunlit_probability=probabilities[1], nonsunlit_probability=probabilities[0])
    # "inner_nearest" means OpenCV's INTER_NEAREST; derive masks before resizing.
    return {name: cv2.resize(mask.astype(np.float32), thermal_shape[::-1],
                             interpolation=cv2.INTER_NEAREST)
            for name, mask in masks.items()}


def statistics(temperatures, weights=None):
    """Population moments, excess kurtosis, quartiles, IQR, P90, P10.

    Empty distributions return NaN; constants have zero std, skewness and kurtosis.
    """
    values = np.asarray(temperatures, dtype=np.float64).ravel()
    weighted = weights is not None
    weights = np.asarray(weights, dtype=np.float64).ravel() if weighted else np.ones_like(values)
    valid = np.isfinite(values) & (weights > 0)
    values, weights = values[valid], weights[valid]
    if not values.size:
        return np.full(10, np.nan)
    mean = np.average(values, weights=weights)
    centered = values - mean
    variance = np.average(centered**2, weights=weights) if np.ptp(values) else 0
    std = np.sqrt(variance)
    skew = np.average((centered / std)**3, weights=weights) if std > 0 else 0
    kurtosis = np.average((centered / std)**4, weights=weights) - 3 if std > 0 else 0
    percentiles = (25, 50, 75, 90, 10)
    if weighted:
        order = np.argsort(values)
        # Inverse weighted empirical CDF; weights describe probability mass, not temperature.
        cdf = np.cumsum(weights[order]) / weights.sum()
        quantiles = values[order][np.searchsorted(cdf, np.array(percentiles) / 100)]
    else:
        quantiles = np.percentile(values, percentiles)
    q1, median, q3, p90, p10 = quantiles
    return np.array((mean, std, skew, kurtosis, q1, median, q3, q3 - q1, p90, p10))


def main():
    output_path.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        original = data_path / "original" / split
        thermal_paths = sorted((original / "thermal").glob("*.npy"))
        if not thermal_paths:
            raise FileNotFoundError(original / "thermal")
        label_temperatures, baseline_temperatures, features = [], [], []
        for path in tqdm(thermal_paths, desc=f"{plant} / {model}: {split}", unit="image"):
            label_path = original / "label" / f"{path.stem}_L.png"
            label = cv2.imread(str(label_path))
            if label is None:
                raise FileNotFoundError(label_path)
            thermal = np.load(path)
            masks = prepare_masks(
                np.all(label == (0, 255, 0), axis=-1),
                np.load(data_path / model / split / "logit" / f"{path.stem}_logits.npy"),
                thermal.shape,
            )
            label_values = thermal[np.isfinite(thermal) & (masks["label"] > 0)]
            label_temperatures.append(label_values.mean() if label_values.size else np.nan)
            # Feature blocks follow the five distributions and ten statistics listed above.
            image_features = np.concatenate((
                statistics(thermal[masks["sunlit_binary"] > 0]),
                statistics(thermal[masks["nonsunlit_binary"] > 0]),
                statistics(thermal, masks["sunlit_probability"]),
                statistics(thermal, masks["nonsunlit_probability"]),
                statistics(thermal),
            ))
            features.append(image_features)
            baseline_temperatures.append(image_features[0])
        # Sorted thermal filenames define image order; correction = label - baseline.
        np.savez_compressed(
            output_path / f"{split}.npz",
            label_mean_temperature=np.asarray(label_temperatures),
            baseline_mean_temperature=np.asarray(baseline_temperatures),
            features=np.asarray(features).T,
        )


if __name__ == "__main__":
    main()
