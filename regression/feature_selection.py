# Exhaustive grouped feature selection, using training CV to select and test to evaluate.
# Search distributions with all statistics, statistics with all distributions, then both.
# Joint subsets use the same selected statistics for every selected distribution.

import csv
from itertools import combinations, product
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


plant = "tomato"
model = "fcdensenet"
root = Path(__file__).resolve().parents[1]
folds = 3
n_jobs = 4
distributions = ("hard_sunlit", "hard_nonsunlit", "soft_sunlit", "soft_nonsunlit", "full_image")
statistics = ("mean", "std", "skewness", "excess_kurtosis", "q1", "median", "q3", "iqr", "p90", "p10")
regressors = {
    "KNN": KNeighborsRegressor(),
}


def subsets(size):
    return [group for count in range(1, size + 1)
            for group in combinations(range(size), count)]


def columns(dist, stats):
    return [10 * d + s for d in dist for s in stats]


def load_split(path):
    with np.load(path) as data:
        x = data["features"].T
        y = (data["label_mean_temperature"].astype(np.float64)
             - data["baseline_mean_temperature"].astype(np.float64))
    if x.shape != (y.size, 50):
        raise ValueError(f"{path}: expected features shaped (50, images)")
    valid = np.isfinite(y)
    if not valid.all():
        print(f"{path.stem}: skipping {(~valid).sum()} missing targets")
    return x[valid], y[valid]


def pipeline(regressor):
    return make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(), clone(regressor))


def score_subset(x, y, regressor, cv, dist, stats):
    # Each CV fold fits its own imputer, scaler and regressor.
    return -cross_val_score(pipeline(regressor), x[:, columns(dist, stats)], y,
                            cv=cv, scoring="neg_mean_squared_error", n_jobs=1,
                            error_score="raise").mean()


def main():
    data_path = root / "results" / plant / model
    if not data_path.is_dir():
        data_path = root / "results" / "feature_extraction" / plant / model
    output_path = root / "results" / "feature_selection" / plant / model
    output_path.mkdir(parents=True, exist_ok=True)
    x, y = load_split(data_path / "train.npz")
    test_x, test_y = load_split(data_path / "test.npz")
    cv = list(KFold(folds, shuffle=True, random_state=42).split(x))
    all_dist, all_stats = tuple(range(5)), tuple(range(10))
    candidates = list(product(subsets(5), subsets(10)))
    # Evaluate every joint candidate once; reuse CV scores for all three searches.
    searches = {
        "all_features": [i for i, pair in enumerate(candidates) if pair == (all_dist, all_stats)],
        "distributions": [i for i, (_, stats) in enumerate(candidates) if stats == all_stats],
        "statistics": [i for i, (dist, _) in enumerate(candidates) if dist == all_dist],
        "joint": range(len(candidates)),
    }
    with (output_path / "cv_scores.csv").open("w", newline="") as scores_file, \
         (output_path / "best_subsets.csv").open("w", newline="") as best_file:
        scores_writer, best_writer = csv.writer(scores_file), csv.writer(best_file)
        fields = ("regressor", "distributions", "statistics", "n_features", "feature_indices_0based")
        scores_writer.writerow((*fields, "cv_rmse"))
        best_writer.writerow(("search", *fields, "cv_rmse", "test_rmse", "test_mae", "test_mbe"))
        for name, regressor in regressors.items():
            with parallel_config(backend="loky", inner_max_num_threads=1):
                results = Parallel(n_jobs=n_jobs, return_as="generator")(
                    delayed(score_subset)(x, y, regressor, cv, dist, stats)
                    for dist, stats in candidates)
                scores = list(tqdm(results, total=len(candidates), desc=f"{model} / {name}"))
            descriptions = []
            for (dist, stats), mse in zip(candidates, scores):
                indices = columns(dist, stats)
                description = (name, "|".join(distributions[d] for d in dist),
                               "|".join(statistics[s] for s in stats), len(indices),
                               "|".join(map(str, indices)))
                descriptions.append(description)
                scores_writer.writerow((*description, np.sqrt(mse)))
            scores_file.flush()
            for search, choices in searches.items():
                best = min(choices, key=lambda i: (scores[i], descriptions[i][3]))
                indices = columns(*candidates[best])
                # Saved pipelines accept all 50 input features and select their own columns.
                fitted = make_pipeline(ColumnTransformer([("selected", "passthrough", indices)]),
                                       pipeline(regressor))
                fitted.fit(x, y)
                error = fitted.predict(test_x) - test_y
                test_rmse = np.sqrt(np.mean(error**2))
                best_writer.writerow((search, *descriptions[best], np.sqrt(scores[best]),
                                      test_rmse, np.mean(np.abs(error)), np.mean(error)))
                best_file.flush()
                joblib.dump(fitted, output_path / f"{name.lower()}_{search}.joblib")
                print(f"{name} / {search}: {len(indices)} features, "
                      f"CV RMSE={np.sqrt(scores[best]):.4f}, test RMSE={test_rmse:.4f}", flush=True)


if __name__ == "__main__":
    main()
