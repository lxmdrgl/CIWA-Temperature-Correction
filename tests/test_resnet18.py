import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torchvision.models import resnet18, resnet34, resnet50

from evaluate import evaluate_checkpoint
from models.resnet18 import ResNet18Regressor
from models.resnet34 import ResNet34Regressor
from models.resnet50 import ResNet50Regressor


class ResNet18Tests(unittest.TestCase):
    model_type = ResNet18Regressor
    model_name = "resnet18"
    builder = staticmethod(resnet18)

    def test_thermal_normalization_preserves_masks_and_temperature_offsets(self):
        model = self.model_type(in_channels=3, pretrained=False)
        train = torch.zeros(2, 3, 4, 4)
        train[0, 0] = 10.
        train[1, 0] = 20.
        train[:, 1:] = 0.75
        model.fit_normalization(train)
        self.assertEqual(model.input_mean.item(), 15.)
        self.assertEqual(model.input_std.item(), 5.)
        model.backbone = torch.nn.Identity()
        normalized = model(train)
        torch.testing.assert_close(normalized[:, 1:], train[:, 1:])
        torch.testing.assert_close(normalized[:, 0], (train[:, 0] - 15.) / 5.)
        shifted = train.clone()
        shifted[:, 0] += 5.
        torch.testing.assert_close(model(shifted)[:, 0], normalized[:, 0] + 1.)

    def test_visual_input_is_unchanged(self):
        model = self.model_type(image="visual", pretrained=False)
        model.backbone = torch.nn.Identity()
        bgr = torch.tensor([0., 127.5, 255.]).view(1, 3, 1, 1)
        torch.testing.assert_close(model(bgr), bgr)

    def test_pretrained_thermal_adaptation_and_frozen_batchnorm(self):
        backbone = self.builder(weights=None)
        rgb_weights = backbone.conv1.weight.detach().clone()
        with patch(f"models.{self.model_name}.{self.model_name}", return_value=backbone) as build:
            model = self.model_type(in_channels=2, pretrained=True, freeze_backbone=True)
        self.assertIsNotNone(build.call_args.kwargs["weights"])
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
        model = self.model_type(in_channels=1, pretrained=False)
        with self.assertRaisesRegex(RuntimeError, "Fit normalization"):
            model(torch.zeros(2, 1, 32, 32))
        model.fit_normalization(torch.ones(2, 1, 32, 32))
        self.assertGreater(model.input_std.item(), 0.)

    def test_mask_initialization_preserves_temperature_and_default_weights(self):
        for pretrained in (False, True):
            for channels in (1, 2, 3):
                with self.subTest(pretrained=pretrained, channels=channels):
                    initial_weights = {}
                    for mask_init in ("default", "zero"):
                        torch.manual_seed(42)
                        backbone = self.builder(weights=None)
                        rgb_weights = backbone.conv1.weight.detach().clone()
                        with patch(f"models.{self.model_name}.{self.model_name}",
                                   return_value=backbone):
                            model = self.model_type(in_channels=channels, pretrained=pretrained,
                                                    mask_init=mask_init)
                        initial_weights[mask_init] = model.backbone.conv1.weight.detach().clone()
                        if pretrained:
                            torch.testing.assert_close(initial_weights[mask_init][:, :1],
                                                       rgb_weights.sum(dim=1, keepdim=True))
                    torch.testing.assert_close(initial_weights["zero"][:, :1],
                                               initial_weights["default"][:, :1])
                    self.assertEqual(torch.count_nonzero(initial_weights["zero"][:, 1:]).item(), 0)
                    if channels > 1:
                        self.assertGreater(torch.count_nonzero(initial_weights["default"][:, 1:]).item(), 0)

    def test_invalid_mask_initialization_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mask_init"):
            self.model_type(pretrained=False, mask_init="invalid")

    def test_thermal_mapping_transfers_weights_to_every_channel(self):
        for channels in (1, 2, 3):
            for mask_init in ("zero", "default"):
                with self.subTest(channels=channels, mask_init=mask_init):
                    backbone = self.builder(weights=None)
                    rgb = backbone.conv1.weight.detach().clone()
                    with patch(f"models.{self.model_name}.{self.model_name}", return_value=backbone):
                        model = self.model_type(in_channels=channels, pretrained=True,
                                                mapping="thermal", mask_init=mask_init)
                    expected = rgb.sum(dim=1, keepdim=True).expand(-1, channels, -1, -1) / channels
                    torch.testing.assert_close(model.backbone.conv1.weight, expected)
                    self.assertEqual(model.input_mean.shape[1], channels)

    def test_conv1_freezing_and_optimizer_learning_rates_are_independent(self):
        for freeze_backbone in (False, True):
            for freeze_conv1 in (False, True):
                with self.subTest(backbone=freeze_backbone, conv1=freeze_conv1):
                    with patch(f"models.{self.model_name}.{self.model_name}",
                               return_value=self.builder(weights=None)):
                        model = self.model_type(pretrained=True, freeze_backbone=freeze_backbone,
                                                freeze_conv1=freeze_conv1, normalize="none")
                    model.train()
                    self.assertEqual(model.backbone.conv1.weight.requires_grad, not freeze_conv1)
                    self.assertEqual(model.backbone.layer1[0].conv1.weight.requires_grad,
                                     not freeze_backbone)
                    self.assertEqual(model.backbone.bn1.training, not freeze_backbone)
                    optimizer = torch.optim.SGD(model.optimizer_parameter_groups(0.0001), lr=0.01)
                    grouped = [p for group in optimizer.param_groups for p in group["params"]]
                    self.assertEqual(len(grouped), len({id(p) for p in grouped}))
                    self.assertEqual({id(p) for p in grouped},
                                     {id(p) for p in model.parameters() if p.requires_grad})
                    conv = model.backbone.conv1.weight
                    head = model.backbone.fc.weight
                    before_conv, before_head = conv.detach().clone(), head.detach().clone()
                    if conv.requires_grad:
                        conv.grad = torch.ones_like(conv)
                    head.grad = torch.ones_like(head)
                    optimizer.step()
                    torch.testing.assert_close(conv, before_conv if freeze_conv1 else before_conv - 0.0001)
                    torch.testing.assert_close(head, before_head - 0.01)

    def test_saved_normalization_is_used_in_evaluation_for_both_targets(self):
        model = self.model_type(in_channels=3, pretrained=False)
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
            "train": {"model": self.model_name, "batch_size": 2, "seed": 0, "normalize": "temp",
                      "learning_rate": 0.001},
            "transfer": {"freeze_conv1": True, "lr_conv1": 0.0001},
        })
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "resnet18.pt"
            for target in ("direct", "correction"):
                cfg.dataset.target = target
                state = model.state_dict()
                saved_config = OmegaConf.to_container(cfg)
                if target == "correction":
                    # Existing checkpoints must still evaluate after the config move.
                    saved_config["train"]["transfer"] = saved_config.pop("transfer")
                torch.save({"config": saved_config, "epoch": 1,
                            "model_state_dict": state}, checkpoint)
                with patch("evaluate.load_dataset", return_value=(data, torch.zeros(2, 1), metadata)), patch(
                    "torch.hub.load_state_dict_from_url", side_effect=AssertionError("Unexpected download")
                ):
                    rows = evaluate_checkpoint(checkpoint, output=Path(directory) / "metrics.csv",
                                               include_baseline=False)
                self.assertEqual(rows[0]["model"], self.model_name)
                self.assertEqual(rows[0]["mae"], 0. if target == "correction" else 17.5)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["kind"], "model")
                self.assertEqual(list(rows[0])[-2:], ["config", "checkpoint"])
            restored = self.model_type(pretrained=False)
            restored.load_state_dict(torch.load(checkpoint, weights_only=True)["model_state_dict"])
            self.assertEqual(restored.input_mean.item(), 15.)
            self.assertEqual(restored.input_std.item(), 5.)
            self.assertTrue(restored.normalization_fitted.item())


class ResNet34Tests(ResNet18Tests):
    model_type = ResNet34Regressor
    model_name = "resnet34"
    builder = staticmethod(resnet34)


class ResNet50Tests(ResNet18Tests):
    model_type = ResNet50Regressor
    model_name = "resnet50"
    builder = staticmethod(resnet50)


if __name__ == "__main__":
    unittest.main()
