import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from evaluate import calculate_metrics, evaluate_checkpoint, run_evaluation
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

    def test_correction_reference_is_reconstructed_from_baseline(self):
        class ExampleDataset:
            thermal_images = [np.array([[10., 20.], [30., 40.]])]
            visual_hard_masks = [np.array([[1, 1], [0, 0]])]

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return np.zeros((2, 2, 3)), 2.0

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
            with self.assertRaisesRegex(ValueError, "Choose mlp, cnn, or resnet18"):
                main.__wrapped__(cfg)
        load.assert_not_called()

    def test_config_selects_all_or_single_checkpoint_and_baseline(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("b.pt", "a.pt", "ignored.txt"):
                (root / name).touch()
            (root / "directory.pt").mkdir()
            cfg = OmegaConf.create({"evaluate": {
                "root": directory, "split": "valid", "checkpoint": "all",
                "name": None, "baseline": False, "output": str(root / "metrics.csv"),
                "legacy_config": None,
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

            cfg.evaluate.checkpoint = "single"
            cfg.evaluate.name = "b.pt"
            cfg.evaluate.baseline = True
            with patch("evaluate.evaluate_checkpoint", return_value=[]) as evaluate:
                run_evaluation(cfg)
                evaluate.assert_called_once()
                self.assertEqual(evaluate.call_args.args[0], root / "b.pt")
                self.assertTrue(evaluate.call_args.kwargs["include_baseline"])

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
            "train": {"model": "mlp", "batch_size": 2, "seed": 7},
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
                ), patch("evaluate.MLPRegressor", return_value=ConstantModel()):
                    # Saved config takes precedence over the optional legacy fallback.
                    rows = evaluate_checkpoint(
                        checkpoint, config_path="unused-legacy-config.yaml",
                        output=Path(directory) / "metrics.csv",
                        include_baseline=include_baseline,
                    )
                self.assertEqual(len(rows), 2 if include_baseline else 1)
                self.assertEqual(rows[0]["kind"], "model")
                self.assertEqual(rows[0]["rmse"], 0.)
                self.assertEqual(rows[0]["seed"], 7)
                if include_baseline:
                    self.assertEqual(rows[1]["kind"], "baseline")
                    self.assertEqual(rows[1]["rmse"], 2.)


if __name__ == "__main__":
    unittest.main()
