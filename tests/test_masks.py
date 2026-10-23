import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from utils.data_pipeline import load_dataset
from utils.dataset import TemperatureDataset


class MaskTests(unittest.TestCase):
    def test_mask_selection_changes_only_input_channels(self):
        thermal = np.array([[10., 20.], [30., 40.]], dtype=np.float32)
        label = np.array([[0., 1.], [0., 1.]], dtype=np.float32)
        hard = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
        soft = np.array([[0.9, 0.2], [0.3, 0.8]], dtype=np.float32)
        source = ([thermal], [np.zeros((2, 2, 3), dtype=np.uint8)],
                  [label], [hard], [soft])
        cfg = OmegaConf.create({"dataset": {
            "root": "unused", "crop": "tomato", "model": "fcdensenet",
            "image": "thermal", "mask": "none", "interpolation": "nearest",
            "target": "correction",
        }})
        masks = {"none": [], "hard": [hard], "soft": [soft], "both": [hard, soft]}
        with patch.object(TemperatureDataset, "get_data", return_value=source):
            for mode, selected in masks.items():
                for target in ("direct", "correction"):
                    with self.subTest(mask=mode, target=target):
                        cfg.dataset.mask, cfg.dataset.target = mode, target
                        data, targets, metadata = load_dataset(cfg, return_metadata=True)
                        expected = np.stack([thermal] + selected)
                        np.testing.assert_array_equal(data[0].numpy(), expected)
                        self.assertEqual(targets.item(), 30. if target == "direct" else 5.)
                        self.assertEqual(metadata["baseline"].item(), 25.)
                        self.assertEqual(metadata["reference"].item(), 30.)

    def test_all_is_not_a_mask_option(self):
        with patch.object(TemperatureDataset, "get_data") as load:
            with self.assertRaisesRegex(ValueError, "none, hard, soft, or both"):
                TemperatureDataset("unused", "train", "tomato", "fcdensenet",
                                   "thermal", "all", "nearest", "direct")
            load.assert_not_called()

    def test_thermal_mapping_preserves_temperature_and_targets(self):
        thermal = np.array([[10., 20.], [30., 40.]], dtype=np.float32)
        label = np.array([[0., 1.], [0., 1.]], dtype=np.float32)
        hard = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
        soft = np.array([[1., 0.], [0.3, 0.8]], dtype=np.float32)
        visual = np.ones((2, 2, 3), dtype=np.uint8)
        source = ([thermal], [visual], [label], [hard], [soft])
        cfg = OmegaConf.create({"dataset": {
            "root": "unused", "crop": "tomato", "model": "fcdensenet",
            "image": "thermal", "mask": "both", "interpolation": "nearest",
            "target": "correction", "mapping": "thermal", "mask_fill": -5.,
        }})
        with patch.object(TemperatureDataset, "get_data", return_value=source):
            for mode, masks in {"none": [], "hard": [hard], "soft": [soft],
                                "both": [hard, soft]}.items():
                for fill in (0., -5.):
                    with self.subTest(mask=mode, fill=fill):
                        cfg.dataset.mask, cfg.dataset.mask_fill = mode, fill
                        data, targets, metadata = load_dataset(cfg, return_metadata=True)
                        expected = [thermal] + [np.where(mask > 0, thermal * mask, fill)
                                                for mask in masks]
                        np.testing.assert_array_equal(data[0].numpy(), np.stack(expected))
                        self.assertEqual(targets.item(), 5.)
                        self.assertEqual(metadata["baseline"].item(), 25.)
                        self.assertEqual(metadata["reference"].item(), 30.)
            cfg.dataset.image = "visual"
            data, _ = load_dataset(cfg)
            np.testing.assert_array_equal(data[0].permute(1, 2, 0).numpy(), visual)


if __name__ == "__main__":
    unittest.main()
