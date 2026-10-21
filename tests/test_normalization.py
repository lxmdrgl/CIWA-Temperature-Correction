import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from evaluate import calculate_metrics, evaluate_checkpoint
from models.cnn import CNNRegressor
from models.mlp import MLPRegressor
from models.normalization import TemperatureNormalizedRegressor
from models.resnet18 import ResNet18Regressor


class NormalizationTests(unittest.TestCase):
    def setUp(self):
        self.data = torch.zeros(2, 3, 32, 32)
        self.data[0, 0] = 10.
        self.data[1, 0] = 20.
        self.data[:, 1] = 1.
        self.data[:, 2] = 0.75

    def test_each_model_normalizes_before_its_first_learned_layer(self):
        models = (
            (MLPRegressor(self.data[0].numel()), "mlp"),
            (CNNRegressor(in_channels=3), "cnn"),
            (ResNet18Regressor(pretrained=False), "resnet18"),
        )
        expected = self.data.clone()
        expected[:, 0] = (expected[:, 0] - 15.) / 5.
        for model, name in models:
            with self.subTest(model=name):
                model.fit_normalization(self.data)
                self.assertEqual(model.thermal_mean.item(), 15.)
                self.assertEqual(model.thermal_std.item(), 5.)
                first = {"mlp": lambda: model.linear_relu_stack[0],
                         "cnn": lambda: model.features[0],
                         "resnet18": lambda: model.backbone.conv1}[name]()
                observed = []
                handle = first.register_forward_pre_hook(
                    lambda module, inputs: observed.append(inputs[0].detach().clone())
                )
                model.eval()
                with torch.inference_mode():
                    self.assertEqual(model(self.data).shape, (2, 1))
                handle.remove()
                torch.testing.assert_close(observed[0], expected.flatten(1) if name == "mlp" else expected)

    def test_invalid_temperatures_and_unfitted_normalization_are_rejected(self):
        model = TemperatureNormalizedRegressor()
        with self.assertRaisesRegex(RuntimeError, "Fit thermal normalization"):
            model.normalize_inputs(self.data)
        for bad in (self.data[:0], torch.full_like(self.data, float("nan")),
                    torch.full_like(self.data, float("inf"))):
            with self.assertRaises(ValueError):
                model.fit_normalization(bad)
        model.fit_normalization(torch.ones_like(self.data))
        self.assertTrue(torch.isfinite(model.normalize_inputs(self.data)).all())

    def test_visual_inputs_are_unchanged_for_mlp_and_cnn(self):
        for model in (MLPRegressor(self.data[0].numel(), image="visual"),
                      CNNRegressor(in_channels=3, image="visual")):
            model.fit_normalization(self.data)
            torch.testing.assert_close(model.normalize_inputs(self.data), self.data)

    def test_evaluation_restores_new_and_legacy_mlp_cnn_checkpoints(self):
        cfg = OmegaConf.create({
            "dataset": {"root": "unused", "crop": "tomato", "model": "fcdensenet",
                        "image": "thermal", "mask": "both",
                        "interpolation": "nearest", "target": "direct"},
            "train": {"model": "mlp", "batch_size": 2, "seed": 0},
        })
        metadata = {"baseline": torch.tensor([[14.], [19.]]),
                    "reference": torch.tensor([[15.], [20.]])}
        with TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            for name in ("mlp", "cnn"):
                for normalized in (False, True):
                    cfg.train.model = name
                    model = (MLPRegressor(self.data[0].numel(), normalize=normalized)
                             if name == "mlp" else CNNRegressor(in_channels=3, normalize=normalized))
                    model.fit_normalization(self.data)
                    model.eval()
                    with torch.inference_mode():
                        raw = model(self.data).double().numpy().reshape(-1)
                    for target in ("direct", "correction"):
                        with self.subTest(model=name, normalized=normalized, target=target):
                            cfg.dataset.target = target
                            checkpoint = Path(directory) / "model.pt"
                            torch.save({"config": OmegaConf.to_container(cfg), "epoch": 1,
                                        "model_state_dict": model.state_dict()}, checkpoint)
                            with patch("evaluate.load_dataset", return_value=(
                                self.data, torch.zeros(2, 1), metadata
                            )):
                                rows = evaluate_checkpoint(checkpoint, output=Path(directory) / "metrics.csv")
                            predicted = raw + metadata["baseline"].numpy().reshape(-1) if target == "correction" else raw
                            expected = calculate_metrics(metadata["reference"].numpy(), predicted)
                            for metric, value in expected.items():
                                np.testing.assert_allclose(rows[0][metric], value, rtol=1e-6)
                            self.assertEqual(rows[1]["mae"], 1.)


if __name__ == "__main__":
    unittest.main()
