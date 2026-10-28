import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.models import resnet18

from evaluate import evaluate_checkpoint
from models.cnn import CNNRegressor
from models.mlp import MLPRegressor
from models.normalization import TemperatureNormalizedRegressor
from models.resnet18 import ResNet18Regressor
from models.resnet34 import ResNet34Regressor
from models.resnet50 import ResNet50Regressor
from utils.data_pipeline import load_dataset
from utils.dataset import TemperatureDataset


class CombinedInputTests(unittest.TestCase):
    def test_rgb_resize_channel_order_masks_and_targets(self):
        thermal = np.array([[10., 20.], [30., 40.]], dtype=np.float32)
        visual = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        label = np.array([[0., 1.], [0., 1.]], dtype=np.float32)
        hard = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
        soft = np.array([[1., 0.], [0.3, 0.8]], dtype=np.float32)
        source = ([thermal], [visual], [label], [hard], [soft])
        rgb = cv2.resize(cv2.cvtColor(visual, cv2.COLOR_BGR2RGB), (2, 2),
                         interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        cfg = OmegaConf.create({"dataset": {
            "root": "unused", "crop": "tomato", "model": "fcdensenet",
            "image": "both", "mask": "both", "interpolation": "nearest",
            "target": "correction",
            "reg_output": "canopy",
        }})
        with patch.object(TemperatureDataset, "get_data", return_value=source):
            for mode, masks in {"none": [], "hard": [hard], "soft": [soft],
                                "both": [hard, soft]}.items():
                for output in ("canopy", "cwsi"):
                    with self.subTest(mask=mode, output=output):
                        cfg.dataset.mask = mode
                        cfg.dataset.reg_output = output
                        cfg.dataset.image = "both"
                        data, labels, metadata = load_dataset(cfg, return_metadata=True)
                        self.assertEqual(data.shape, (1, 4 + len(masks), 2, 2))
                        self.assertEqual(data.dtype, torch.float32)
                        np.testing.assert_array_equal(data[0, :3].numpy(), rgb)
                        np.testing.assert_array_equal(data[0, 3:].numpy(),
                                                      np.stack([thermal] + masks))
                        cfg.dataset.image = "thermal"
                        _, thermal_labels, thermal_metadata = load_dataset(cfg, return_metadata=True)
                        torch.testing.assert_close(labels, thermal_labels)
                        for key in metadata:
                            torch.testing.assert_close(metadata[key], thermal_metadata[key])
            cfg.dataset.image = "visual"
            data, _ = load_dataset(cfg)
            self.assertEqual(data.shape, (1, 3, 4, 6))
            np.testing.assert_array_equal(data[0].permute(1, 2, 0).numpy(),
                                          cv2.cvtColor(visual, cv2.COLOR_BGR2RGB))
            np.testing.assert_array_equal(source[1][0], visual)

    def test_combined_normalization_uses_the_correct_channels(self):
        data = torch.arange(2 * 6 * 4 * 4, dtype=torch.float32).reshape(2, 6, 4, 4)
        for channels in (4, 5, 6):
            train = data[:, :channels]
            for mode in ("none", "temp", "all"):
                with self.subTest(channels=channels, normalize=mode):
                    model = TemperatureNormalizedRegressor(image="both", in_channels=channels,
                                                           normalize=mode)
                    model.fit_normalization(train)
                    expected = train.clone()
                    rgb = train[:, :3] / 255.0
                    mean = rgb.mean(dim=(0, 2, 3), keepdim=True)
                    std = rgb.std(dim=(0, 2, 3), correction=0, keepdim=True)
                    expected[:, :3] = (rgb - mean) / std
                    if mode != "none":
                        selected = (slice(3, channels) if mode == "all" else
                                    slice(3, 4))
                        values = expected[:, selected]
                        mean = values.mean(dim=(0, 2, 3), keepdim=True)
                        std = values.std(dim=(0, 2, 3), correction=0, keepdim=True)
                        expected[:, selected] = (values - mean) / std
                    torch.testing.assert_close(model.normalize_inputs(train), expected)

    def test_combined_pretrained_filters_preserve_the_original_channel_sum(self):
        for channels in (4, 5, 6):
            backbone = resnet18(weights=None)
            rgb_weights = backbone.conv1.weight.detach().clone()
            with patch("models.resnet18.resnet18", return_value=backbone):
                model = ResNet18Regressor(image="both", in_channels=channels,
                                         pretrained=True)
            actual = model.backbone.conv1.weight
            torch.testing.assert_close(actual[:, :3], rgb_weights * 0.75)
            thermal = rgb_weights.sum(dim=1, keepdim=True)
            torch.testing.assert_close(actual[:, 3:4], thermal / 4.0)
            torch.testing.assert_close(actual[:, :4].sum(dim=1, keepdim=True), thermal)
            self.assertEqual(torch.count_nonzero(actual[:, 4:]).item(), 0)

    def test_all_models_restore_combined_input_checkpoints(self):
        data = torch.rand(2, 6, 32, 32)
        cfg = OmegaConf.create({
            "dataset": {"root": "unused", "crop": "tomato", "model": "fcdensenet",
                        "image": "both", "mask": "both", "interpolation": "nearest",
                        "target": "direct", "reg_output": "cwsi"},
            "train": {"model": "mlp", "batch_size": 2, "seed": 0,
                      "normalize": "temp", "learning_rate": 0.001},
            "transfer": {"freeze_conv1": False, "lr_conv1": 0.0001},
        })
        builders = {
            "mlp": lambda: MLPRegressor(data[0].numel(), in_channels=6, image="both", num_outputs=3),
            "cnn": lambda: CNNRegressor(in_channels=6, image="both", num_outputs=3),
            "resnet18": lambda: ResNet18Regressor(in_channels=6, image="both", num_outputs=3, pretrained=False),
            "resnet34": lambda: ResNet34Regressor(in_channels=6, image="both", num_outputs=3, pretrained=False),
            "resnet50": lambda: ResNet50Regressor(in_channels=6, image="both", num_outputs=3, pretrained=False),
        }
        with TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            for name, build in builders.items():
                with self.subTest(model=name):
                    model = build()
                    model.fit_normalization(data)
                    model.eval()
                    with torch.inference_mode():
                        reference = model(data).double()
                    self.assertEqual(reference.shape, (2, 3))
                    cfg.train.model = name
                    checkpoint = Path(directory) / "model.pt"
                    torch.save({"config": OmegaConf.to_container(cfg), "epoch": 1,
                                "model_state_dict": model.state_dict()}, checkpoint)
                    metadata = {"baseline": torch.zeros_like(reference), "reference": reference}
                    with patch("evaluate.load_dataset", return_value=(data, reference, metadata)):
                        rows = evaluate_checkpoint(checkpoint, include_baseline=False,
                                                  output=Path(directory) / "metrics.csv")
                    self.assertTrue(all(row["mae"] == 0 for row in rows))


if __name__ == "__main__":
    unittest.main()
