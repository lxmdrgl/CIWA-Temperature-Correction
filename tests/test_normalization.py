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
from models.resnet34 import ResNet34Regressor
from models.resnet50 import ResNet50Regressor


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
            (ResNet34Regressor(pretrained=False), "resnet34"),
            (ResNet50Regressor(pretrained=False), "resnet50"),
        )
        expected = self.data.clone()
        expected[:, 0] = (expected[:, 0] - 15.) / 5.
        for model, name in models:
            with self.subTest(model=name):
                model.fit_normalization(self.data)
                self.assertEqual(model.input_mean.item(), 15.)
                self.assertEqual(model.input_std.item(), 5.)
                first = {"mlp": lambda: model.linear_relu_stack[0],
                         "cnn": lambda: model.features[0],
                         "resnet18": lambda: model.backbone.conv1,
                         "resnet34": lambda: model.backbone.conv1,
                         "resnet50": lambda: model.backbone.conv1}[name]()
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
        with self.assertRaisesRegex(RuntimeError, "Fit normalization"):
            model.normalize_inputs(self.data)
        for bad in (self.data[:0], torch.full_like(self.data, float("nan")),
                    torch.full_like(self.data, float("inf"))):
            with self.assertRaises(ValueError):
                model.fit_normalization(bad)
        model.fit_normalization(torch.ones_like(self.data))
        self.assertTrue(torch.isfinite(model.normalize_inputs(self.data)).all())

    def test_visual_inputs_use_training_statistics_for_mlp_and_cnn(self):
        data = torch.arange(2 * 3 * 32 * 32, dtype=torch.float32).reshape(2, 3, 32, 32) % 256
        for model in (MLPRegressor(self.data[0].numel(), image="visual"),
                      CNNRegressor(in_channels=3, image="visual")):
            with self.assertRaisesRegex(RuntimeError, "Fit normalization"):
                model.normalize_inputs(data)
            model.fit_normalization(data)
            rgb = data / 255.0
            mean = rgb.mean(dim=(0, 2, 3), keepdim=True)
            std = rgb.std(dim=(0, 2, 3), correction=0, keepdim=True)
            torch.testing.assert_close(model.rgb_mean, mean)
            torch.testing.assert_close(model.rgb_std, std)
            torch.testing.assert_close(model.normalize_inputs(data), (rgb - mean) / std)
            # Another split reuses the training statistics.
            torch.testing.assert_close(model.normalize_inputs(data + 5),
                                       ((data + 5) / 255.0 - mean) / std)

    def test_rgb_training_statistics_apply_in_every_normalization_mode(self):
        data = torch.tensor([0., 127.5, 255.]).reshape(1, 1, 1, 3).expand(2, 3, 1, 3).clone()
        original = data.clone()
        for mode in ("none", "temp", "all"):
            with self.subTest(normalize=mode):
                model = TemperatureNormalizedRegressor(image="visual", normalize=mode)
                model.fit_normalization(data)
                scaled = data / 255.0
                mean = scaled.mean(dim=(0, 2, 3), keepdim=True)
                std = scaled.std(dim=(0, 2, 3), correction=0, keepdim=True)
                torch.testing.assert_close(model.rgb_mean, mean)
                torch.testing.assert_close(model.rgb_std, std)
                expected = (scaled - mean) / std
                torch.testing.assert_close(model.normalize_inputs(data), expected)
                torch.testing.assert_close(data, original)

    def test_resnet_rgb_is_not_refitted_or_normalized_twice(self):
        data = torch.arange(2 * 6 * 4 * 4, dtype=torch.float32).reshape(2, 6, 4, 4)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        for image in ("visual", "both"):
            train = data[:, :3] if image == "visual" else data
            for mode in ("none", "temp", "all"):
                model = TemperatureNormalizedRegressor(image=image, in_channels=train.shape[1],
                    normalize=mode, rgb_normalization="imagenet")
                model.fit_normalization(train)
                actual = model.normalize_inputs(train)
                torch.testing.assert_close(actual[:, :3], (train[:, :3] / 255.0 - mean) / std)
                torch.testing.assert_close(model.rgb_mean, mean)
                torch.testing.assert_close(model.rgb_std, std)
                if image == "both" and mode == "temp":
                    torch.testing.assert_close(actual[:, 4:], train[:, 4:])

    def test_constant_rgb_channels_have_finite_normalization(self):
        data = torch.full((2, 3, 4, 4), 127.5)
        model = TemperatureNormalizedRegressor(image="visual")
        model.fit_normalization(data)
        torch.testing.assert_close(model.normalize_inputs(data), torch.zeros_like(data))

    def test_all_standardizes_each_channel_using_fixed_training_statistics(self):
        for channels in (1, 2, 3):
            with self.subTest(channels=channels):
                train = self.data[:, :channels].clone()
                if channels > 1:
                    train[0, 1] = 0.
                model = TemperatureNormalizedRegressor(normalize="all", in_channels=channels)
                model.fit_normalization(train)
                mean = train.mean(dim=(0, 2, 3), keepdim=True)
                std = train.std(dim=(0, 2, 3), correction=0, keepdim=True).clamp_min(1e-6)
                torch.testing.assert_close(model.input_mean, mean)
                torch.testing.assert_close(model.input_std, std)
                torch.testing.assert_close(model.normalize_inputs(train), (train - mean) / std)
                # A different split must reuse these values, not fit its own statistics.
                valid = train + 2.
                torch.testing.assert_close(model.normalize_inputs(valid), (valid - mean) / std)
                torch.testing.assert_close(model.input_mean, mean)

    def test_none_returns_raw_inputs_without_fitting(self):
        model = TemperatureNormalizedRegressor(normalize="none")
        model.fit_normalization(self.data)
        torch.testing.assert_close(model.normalize_inputs(self.data), self.data)
        self.assertEqual(dict(model.named_buffers()), {})


    def test_invalid_normalization_modes_are_rejected(self):
        for mode in (True, False, None, "invalid"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "none, temp, or all"):
                TemperatureNormalizedRegressor(normalize=mode)

    def test_evaluation_restores_each_models_configured_normalization(self):
        cfg = OmegaConf.create({
            "dataset": {"root": "unused", "crop": "tomato", "model": "fcdensenet",
                        "image": "thermal", "mask": "both",
                        "interpolation": "nearest", "target": "direct"},
            "train": {"model": "mlp", "batch_size": 2, "seed": 0, "learning_rate": 0.001},
            "transfer": {"freeze_conv1": False, "lr_conv1": 0.0001},
        })
        metadata = {"baseline": torch.tensor([[14.], [19.]]),
                    "reference": torch.tensor([[15.], [20.]])}
        with TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            for name in ("mlp", "cnn", "resnet18", "resnet34", "resnet50"):
                for normalized in ('none', 'temp', 'all'):
                    cfg.train.model = name
                    cfg.train.normalize = normalized
                    if name == "mlp":
                        model = MLPRegressor(self.data[0].numel(), normalize=normalized)
                    elif name == "cnn":
                        model = CNNRegressor(in_channels=3, normalize=normalized)
                    else:
                        regressors = {"resnet18": ResNet18Regressor,
                                      "resnet34": ResNet34Regressor,
                                      "resnet50": ResNet50Regressor}
                        model = regressors[name](in_channels=3, pretrained=False,
                                                 normalize=normalized)
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
                                rows = evaluate_checkpoint(checkpoint, output=Path(directory) / "metrics.csv",
                                                           include_baseline=False)
                            predicted = raw + metadata["baseline"].numpy().reshape(-1) if target == "correction" else raw
                            expected = calculate_metrics(metadata["reference"].numpy(), predicted)
                            for metric, value in expected.items():
                                np.testing.assert_allclose(rows[0][metric], value, rtol=1e-6)
                            self.assertEqual(len(rows), 1)
                            self.assertEqual(rows[0]["kind"], "model")


if __name__ == "__main__":
    unittest.main()
