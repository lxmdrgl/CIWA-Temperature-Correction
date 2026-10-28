import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from evaluate import calculate_metrics, evaluate_checkpoint, run_evaluation, write_metrics
from main import main
from utils.data_pipeline import load_dataset


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_bias_sign(self):
        result = calculate_metrics([2, 4, 6], [3, 3, 8])
        self.assertAlmostEqual(result["r2"], 0.25)
        self.assertAlmostEqual(result["mse"], 2)
        self.assertAlmostEqual(result["rmse"], np.sqrt(2))
        self.assertAlmostEqual(result["mae"], 4 / 3)
        self.assertAlmostEqual(result["mbe"], 2 / 3)

    def test_constant_reference_r2_is_undefined(self):
        result = calculate_metrics([5, 5], [5, 5])
        self.assertTrue(np.isnan(result["r2"]))
        self.assertEqual(result["mse"], 0)

    def test_invalid_predictions_are_rejected(self):
        for reference, prediction in (([], []), ([1, 2], [1]), ([1], [np.nan])):
            with self.subTest(prediction=prediction), self.assertRaises(ValueError):
                calculate_metrics(reference, prediction)

    def test_checkpoint_requires_saved_training_configuration(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            for config in ({}, {"config": None}):
                torch.save({**config, "epoch": 1, "model_state_dict": {}}, checkpoint)
                with patch("evaluate.load_dataset") as load:
                    with self.assertRaisesRegex(ValueError, "saved training configuration"):
                        evaluate_checkpoint(checkpoint)
                load.assert_not_called()

    def test_correction_reference_is_reconstructed_from_baseline(self):
        class ExampleDataset:
            thermal_images = [np.array([[10., 20.], [30., 40.]])]
            visual_hard_masks = [np.array([[1, 1], [0, 0]])]

            def __len__(self):
                return 1

            def get_sample(self, index):
                return np.zeros((2, 2, 3)), 2.0, {"baseline": 15.0}

        cfg = SimpleNamespace(dataset=SimpleNamespace(
            root="unused", crop="tomato", model="fcdensenet", image="thermal",
            mask="both", interpolation="nearest", target="correction",
        ))
        with patch("utils.data_pipeline.TemperatureDataset", return_value=ExampleDataset()):
            data, labels, metadata = load_dataset(cfg, return_metadata=True)
        self.assertEqual(data.shape, (1, 3, 2, 2))
        self.assertEqual(data.dtype, torch.float32)
        self.assertEqual(labels.item(), 2)
        self.assertEqual(metadata["baseline"].item(), 15)
        self.assertEqual(metadata["reference"].item(), 17)

    def test_training_rejects_unsupported_models_before_loading_data(self):
        cfg = OmegaConf.create({"train": {"model": "unsupported"}})
        with patch("main.load_dataset") as load:
            with self.assertRaisesRegex(ValueError, "Choose mlp, cnn, resnet18, resnet34, or resnet50"):
                main.__wrapped__(cfg)
        load.assert_not_called()

    def test_training_passes_normalization_and_transfer_settings_to_models(self):
        cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/config.yaml")
        cfg.dataset.image = "thermal"
        cfg.dataset.reg_output = "canopy"
        cfg.transfer.pretrained = False
        cfg.transfer.freeze_backbone = False
        data = torch.zeros(2, 3, 32, 32)
        labels = torch.zeros(2, 1)
        constructors = {"mlp": "main.MLPRegressor", "cnn": "main.CNNRegressor",
                        "resnet18": "models.resnet18.ResNet18Regressor",
                        "resnet34": "models.resnet34.ResNet34Regressor",
                        "resnet50": "models.resnet50.ResNet50Regressor"}
        for name, constructor in constructors.items():
            for normalize in ('none', 'temp', 'all'):
                with self.subTest(model=name, normalize=normalize):
                    cfg.train.model = name
                    cfg.train.normalize = normalize
                    with patch("main.load_dataset", return_value=(data, labels)), patch(
                        constructor, side_effect=RuntimeError("constructor reached")
                    ) as build, patch("torch.cuda.is_available", return_value=False):
                        with self.assertRaisesRegex(RuntimeError, "constructor reached"):
                            main.__wrapped__(cfg)
                    self.assertEqual(build.call_args.kwargs["normalize"], normalize)
                    if name.startswith("resnet"):
                        self.assertFalse(build.call_args.kwargs["pretrained"])
                        self.assertFalse(build.call_args.kwargs["freeze_backbone"])
                        self.assertEqual(build.call_args.kwargs["freeze_conv1"],
                                         cfg.transfer.freeze_conv1)
                        self.assertEqual(build.call_args.kwargs["mask_init"],
                                         cfg.transfer.mask_init)

    def test_config_selects_all_or_single_checkpoint_and_baseline(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("b.pt", "a.pt", "ignored.txt"):
                (root / name).touch()
            (root / "directory.pt").mkdir()
            cfg = OmegaConf.create({"evaluate": {
                "root": directory, "split": "valid", "checkpoint": "all",
                "name": None, "baseline": False, "output": str(root / "metrics.csv"),
            }})
            with patch("evaluate.evaluate_checkpoint", return_value=[{"kind": "model"}]) as evaluate:
                rows = run_evaluation(cfg)
                self.assertEqual(len(rows), 2)
                self.assertEqual([call.args[0].name for call in evaluate.call_args_list],
                                 ["a.pt", "b.pt"])
                for call in evaluate.call_args_list:
                    self.assertEqual(call.kwargs["split"], "valid")
                    self.assertFalse(call.kwargs["include_baseline"])
                    self.assertEqual(call.kwargs["output"], str(root / "metrics.csv"))
                    self.assertEqual(call.kwargs["metric"], "all")

            cfg.evaluate.checkpoint = "single"
            cfg.evaluate.name = "b.pt"
            cfg.evaluate.baseline = True
            cfg.evaluate.metric = "avg"
            with patch("evaluate.evaluate_checkpoint", return_value=[]) as evaluate:
                run_evaluation(cfg)
                evaluate.assert_called_once()
                self.assertEqual(evaluate.call_args.args[0], root / "b.pt")
                self.assertTrue(evaluate.call_args.kwargs["include_baseline"])
                self.assertEqual(evaluate.call_args.kwargs["metric"], "avg")

    def test_avg_metrics_pool_all_temperatures_and_preserve_sample_count(self):
        cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/config.yaml")
        cfg.dataset.reg_output = "cwsi"
        reference = np.array([[10., 20., 30.], [40., 50., 60.]])
        prediction = reference + np.array([[1., -2., 3.], [4., -5., 6.]])
        with TemporaryDirectory() as directory:
            output = Path(directory) / "metrics.csv"
            separate = write_metrics(cfg, "test", reference, [("model", prediction)], output)
            combined = write_metrics(cfg, "test", reference, [("model", prediction)],
                                     output, metric="avg")
            self.assertEqual(len(separate), 3)
            self.assertEqual(len(combined), 1)
            row = combined[0]
            self.assertEqual(row["output"], "avg")
            self.assertEqual(row["samples"], 2)
            self.assertAlmostEqual(row["mse"], 91 / 6)
            self.assertAlmostEqual(row["rmse"], np.sqrt(91 / 6))
            self.assertAlmostEqual(row["mae"], 3.5)
            self.assertAlmostEqual(row["mbe"], 7 / 6)
            self.assertAlmostEqual(row["r2"], 1 - 91 / 1750)
            self.assertNotAlmostEqual(row["rmse"], np.mean([r["rmse"] for r in separate]))
            cfg.dataset.reg_output = "canopy"
            canopy = write_metrics(cfg, "test", reference[:, :1],
                [("model", prediction[:, :1])], output, metric="avg")
            self.assertEqual(len(canopy), 1)
            self.assertAlmostEqual(canopy[0]["mse"], 8.5)

    def test_avg_checkpoint_evaluation_combines_reconstructed_corrections(self):
        class ConstantModel(torch.nn.Module):
            def forward(self, inputs):
                return torch.tensor([1., -2., 3.], device=inputs.device).expand(len(inputs), -1)

        cfg = OmegaConf.create({
            "dataset": {"root": "unused", "crop": "tomato", "model": "fcdensenet",
                        "image": "thermal", "mask": "hard", "interpolation": "nearest",
                        "target": "correction", "reg_output": "cwsi"},
            "train": {"model": "mlp", "batch_size": 2, "seed": 0,
                      "normalize": "none", "learning_rate": 0.001},
        })
        baseline = torch.tensor([[10., 20., 5.], [30., 40., 25.]])
        reference = baseline + torch.tensor([2., 4., 6.])
        data = torch.zeros(2, 1, 4, 4)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save({"config": OmegaConf.to_container(cfg), "epoch": 1,
                        "model_state_dict": {}}, checkpoint)
            for baseline_only in (True, False):
                with patch("evaluate.load_dataset", return_value=(data, reference - baseline,
                    {"baseline": baseline, "reference": reference})), patch(
                    "evaluate.MLPRegressor", return_value=ConstantModel()
                ):
                    rows = evaluate_checkpoint(checkpoint, include_baseline=baseline_only,
                        output=Path(directory) / "metrics.csv", metric="avg")
                prediction = baseline if baseline_only else baseline + torch.tensor([1., -2., 3.])
                expected = calculate_metrics(reference.numpy(), prediction.numpy())
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["output"], "avg")
                for name, value in expected.items():
                    self.assertAlmostEqual(rows[0][name], value)

    def test_invalid_metric_selection_is_rejected_before_loading(self):
        with patch("evaluate.torch.load") as load:
            with self.assertRaisesRegex(ValueError, "evaluate.metric"):
                evaluate_checkpoint("unused.pt", metric="invalid")
            load.assert_not_called()
        cfg = OmegaConf.create({"evaluate": {"split": "test", "baseline": False,
                                            "metric": "invalid"}})
        with self.assertRaisesRegex(ValueError, "evaluate.metric"):
            run_evaluation(cfg)
        with self.assertRaisesRegex(ValueError, "evaluate.metric"):
            write_metrics(cfg, "test", [1.], [("model", [1.])], "unused.csv", metric="invalid")

    def test_invalid_evaluation_selection_is_rejected(self):
        with TemporaryDirectory() as directory:
            cfg = OmegaConf.create({"evaluate": {
                "root": directory, "split": "test", "checkpoint": "all",
                "name": None, "baseline": True,
            }})
            with self.assertRaisesRegex(ValueError, "No .pt checkpoints"):
                run_evaluation(cfg)
            cfg.evaluate.checkpoint = "single"
            with self.assertRaisesRegex(ValueError, "evaluate.name"):
                run_evaluation(cfg)
            cfg.evaluate.name = "missing.pt"
            with self.assertRaises(FileNotFoundError):
                run_evaluation(cfg)
            cfg.evaluate.checkpoint = "invalid"
            with self.assertRaisesRegex(ValueError, "all or single"):
                run_evaluation(cfg)
            cfg.evaluate.split = "invalid"
            with self.assertRaisesRegex(ValueError, "evaluate.split"):
                run_evaluation(cfg)

    def test_checkpoint_baseline_toggle_preserves_correction_predictions(self):
        class ConstantModel(torch.nn.Module):
            def forward(self, inputs):
                return torch.full((len(inputs), 1), 2., device=inputs.device)

        cfg = OmegaConf.create({
            "dataset": {"root": "unused", "crop": "tomato", "model": "fcdensenet",
                        "image": "thermal", "mask": "both",
                        "interpolation": "nearest", "target": "correction"},
            "train": {"model": "mlp", "batch_size": 2, "seed": 7, "normalize": "none",
                      "learning_rate": 0.001},
        })
        data = torch.zeros(2, 3, 2, 2)
        labels = torch.tensor([[2.], [2.]])
        metadata = {"baseline": torch.tensor([[15.], [20.]]),
                    "reference": torch.tensor([[17.], [22.]])}
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save({"config": OmegaConf.to_container(cfg), "epoch": 3,
                        "model_state_dict": {}}, checkpoint)
            for include_baseline in (True, False):
                with self.subTest(include_baseline=include_baseline), patch(
                    "evaluate.load_dataset", return_value=(data, labels, metadata)
                ), patch("evaluate.MLPRegressor", return_value=ConstantModel()) as build:
                    rows = evaluate_checkpoint(
                        checkpoint,
                        output=Path(directory) / "metrics.csv",
                        include_baseline=include_baseline,
                    )
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["kind"], "baseline" if include_baseline else "model")
                self.assertEqual(rows[0]["rmse"], 2. if include_baseline else 0.)
                self.assertEqual(rows[0]["seed"], 7)
                if include_baseline:
                    build.assert_not_called()
                else:
                    build.assert_called_once()


    def test_result_column_selection_and_learning_rates(self):
        cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/config.yaml")
        cfg.dataset.reg_output = "canopy"
        cfg.train.model = "resnet50"
        cfg.train.learning_rate = 0.001
        cfg.transfer.lr_conv1 = 0.0001
        cfg.transfer.mask_init = "default"
        required = ["epoch", "model", "crop", "seg model", "image"]
        dataset_columns = ["mask", "interpolation", "target"]
        train_columns = ["normalize", "mask_init", "conv1", "fc"]
        with TemporaryDirectory() as directory:
            for column in ("dataset", "train", "all"):
                for frozen in (True, False):
                    with self.subTest(column=column, frozen=frozen):
                        cfg.transfer.freeze_conv1 = frozen
                        rows = write_metrics(cfg, "valid", [1., 2.], [("model", [1., 2.])],
                                             Path(directory) / f"{column}_{frozen}.csv", column=column)
                        expected = required + (dataset_columns if column != "train" else [])
                        expected += train_columns if column != "dataset" else []
                        self.assertEqual(list(rows[0])[:len(expected)], expected)
                        for name in dataset_columns:
                            self.assertEqual(name in rows[0], column != "train")
                        for name in train_columns:
                            self.assertEqual(name in rows[0], column != "dataset")
                        if column != "dataset":
                            self.assertEqual(rows[0]["mask_init"], "default")
                            self.assertEqual(rows[0]["conv1"], 0 if frozen else 0.0001)
                            self.assertEqual(rows[0]["fc"], 0.001)
            with self.assertRaisesRegex(ValueError, "evaluate.column"):
                write_metrics(cfg, "valid", [1.], [("model", [1.])],
                              Path(directory) / "invalid.csv", column="invalid")


if __name__ == "__main__":
    unittest.main()
