import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torchvision.models import resnet18

from evaluate import evaluate_checkpoint
from models.resnet18 import ResNet18Regressor


class ResNet18Tests(unittest.TestCase):
    def test_thermal_normalization_preserves_masks_and_temperature_offsets(self):
        model = ResNet18Regressor(in_channels=3, pretrained=False)
        train = torch.zeros(2, 3, 4, 4)
        train[0, 0] = 10.
        train[1, 0] = 20.
        train[:, 1:] = 0.75
        model.fit_normalization(train)
        self.assertEqual(model.thermal_mean.item(), 15.)
        self.assertEqual(model.thermal_std.item(), 5.)
        model.backbone = torch.nn.Identity()
        normalized = model(train)
        torch.testing.assert_close(normalized[:, 1:], train[:, 1:])
        torch.testing.assert_close(normalized[:, 0], (train[:, 0] - 15.) / 5.)
        shifted = train.clone()
        shifted[:, 0] += 5.
        torch.testing.assert_close(model(shifted)[:, 0], normalized[:, 0] + 1.)

    def test_visual_input_is_converted_from_bgr_and_normalized(self):
        model = ResNet18Regressor(image="visual", pretrained=False)
        model.backbone = torch.nn.Identity()
        bgr = torch.tensor([0., 127.5, 255.]).view(1, 3, 1, 1)
        expected = (torch.tensor([1., 0.5, 0.]).view(1, 3, 1, 1)
                    - model.rgb_mean) / model.rgb_std
        torch.testing.assert_close(model(bgr), expected)

    def test_pretrained_thermal_adaptation_and_frozen_batchnorm(self):
        backbone = resnet18(weights=None)
        rgb_weights = backbone.conv1.weight.detach().clone()
        with patch("models.resnet18.resnet18", return_value=backbone):
            model = ResNet18Regressor(in_channels=2, pretrained=True, freeze_backbone=True)
        torch.testing.assert_close(model.backbone.conv1.weight[:, :1],
                                   rgb_weights.sum(dim=1, keepdim=True))
        self.assertEqual(torch.count_nonzero(model.backbone.conv1.weight[:, 1:]).item(), 0)
        model.train()
        self.assertFalse(model.backbone.bn1.training)
        self.assertFalse(model.backbone.layer4[0].bn1.training)
        self.assertTrue(model.backbone.fc.weight.requires_grad)
        self.assertTrue(model.backbone.conv1.weight.requires_grad)
        self.assertFalse(model.backbone.layer4[0].conv1.weight.requires_grad)

    def test_unfitted_thermal_inputs_are_rejected(self):
        model = ResNet18Regressor(in_channels=1, pretrained=False)
        with self.assertRaisesRegex(RuntimeError, "Fit thermal normalization"):
            model(torch.zeros(2, 1, 32, 32))
        model.fit_normalization(torch.ones(2, 1, 32, 32))
        self.assertGreater(model.thermal_std.item(), 0.)

    def test_saved_normalization_is_used_in_evaluation_for_both_targets(self):
        model = ResNet18Regressor(in_channels=3, pretrained=False)
        data = torch.zeros(2, 3, 32, 32)
        data[0, 0] = 10.
        data[1, 0] = 20.
        model.fit_normalization(data)
        with torch.no_grad():
            model.backbone.fc.weight.zero_()
            model.backbone.fc.bias.fill_(2.)
        metadata = {"baseline": torch.tensor([[15.], [20.]]),
                    "reference": torch.tensor([[17.], [22.]])}
        cfg = OmegaConf.create({
            "dataset": {"root": "unused", "crop": "tomato", "model": "fcdensenet",
                        "image": "thermal", "mask": "both",
                        "interpolation": "nearest", "target": "correction"},
            "train": {"model": "resnet18", "batch_size": 2, "seed": 0},
        })
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "resnet18.pt"
            for target in ("direct", "correction"):
                cfg.dataset.target = target
                torch.save({"config": OmegaConf.to_container(cfg), "epoch": 1,
                            "model_state_dict": model.state_dict()}, checkpoint)
                with patch("evaluate.load_dataset", return_value=(data, torch.zeros(2, 1), metadata)), patch(
                    "torch.hub.load_state_dict_from_url", side_effect=AssertionError("Unexpected download")
                ):
                    rows = evaluate_checkpoint(checkpoint, output=Path(directory) / "metrics.csv")
                self.assertEqual(rows[0]["model"], "resnet18")
                self.assertEqual(rows[0]["mae"], 0. if target == "correction" else 17.5)
                self.assertEqual(rows[1]["kind"], "baseline")
                self.assertEqual(list(rows[0])[-2:], ["config", "checkpoint"])
            restored = ResNet18Regressor(pretrained=False)
            restored.load_state_dict(torch.load(checkpoint, weights_only=True)["model_state_dict"])
            self.assertEqual(restored.thermal_mean.item(), 15.)
            self.assertEqual(restored.thermal_std.item(), 5.)
            self.assertTrue(restored.thermal_normalization_fitted.item())


if __name__ == "__main__":
    unittest.main()
