import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


plant = "tomato"
model = "fcdensenet"
root = Path(__file__).resolve().parents[1]
data_path = root / "results" / plant / model
if not data_path.is_dir():
    data_path = root / "results" / "feature_extraction" / plant / model
output_path = root / "results" / "training" / plant / model


def load_split(split):
    with np.load(data_path / f"{split}.npz") as data:
        features = data["features"].T  # scikit-learn expects (images, features).
        target = data["label_mean_temperature"] - data["baseline_mean_temperature"]
    valid = np.isfinite(target)
    if not valid.all():
        print(f"{split}: skipping {(~valid).sum()} images with missing target temperatures")
    return features[valid], target[valid]


def main():
    data = {split: load_split(split) for split in ("train", "valid", "test")}
    regressors = {
        "MLP": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000,
                            early_stopping=True, random_state=42),
        "KNN": KNeighborsRegressor(),
        "SVM": SVR(),
    }
    output_path.mkdir(parents=True, exist_ok=True)
    rows = [("baseline", *(np.mean(y**2) for _, y in data.values()))]
    for name, regressor in regressors.items():
        # Fit imputation, scaling and regression only on the training split.
        pipeline = make_pipeline(SimpleImputer(keep_empty_features=True),
                                 StandardScaler(), regressor)
        pipeline.fit(*data["train"])
        scores = [mean_squared_error(y, pipeline.predict(x)) for x, y in data.values()]
        rows.append((name, *scores))
        joblib.dump(pipeline, output_path / f"{name.lower()}.joblib")
    with (output_path / "mse.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("model", "train_mse", "valid_mse", "test_mse"))
        writer.writerows(rows)
    print("Model     Train MSE   Valid MSE    Test MSE")
    for name, *scores in rows:
        print(f"{name:8s}" + "".join(f"{score:12.4f}" for score in scores))


if __name__ == "__main__":
    main()
