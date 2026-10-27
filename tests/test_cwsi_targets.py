import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from utils.data_pipeline import load_dataset
from utils.dataset import TemperatureDataset


class CwsiTargetTests(unittest.TestCase):
    def test_direct_and_correction_targets_keep_three_outputs_per_sample(self):
        thermal = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
        label = np.ones_like(thermal)
        label[0] = 0  # Exclude the coldest ten pixels from the label canopy.
        hard = np.ones_like(thermal)
        source = ([thermal, thermal], [np.zeros((10, 10, 3), dtype=np.uint8)] * 2,
                  [label, label], [hard, hard], [hard, hard])
        cfg = OmegaConf.create({"dataset": {
            "root": "unused", "crop": "tomato", "model": "fcdensenet",
            "image": "thermal", "mask": "none", "interpolation": "nearest",
            "target": "direct", "reg_output": "cwsi", "bound": 5.0,
        }})
        expected = np.array([[55.5, 98., 13.]] * 2)
        baseline = np.array([[50.5, 98., 3.]] * 2)
        with patch.object(TemperatureDataset, "get_data", return_value=source):
            for target in ("direct", "correction"):
                with self.subTest(target=target):
                    cfg.dataset.target = target
                    _, labels, metadata = load_dataset(cfg, return_metadata=True)
                    self.assertEqual(tuple(labels.shape), (2, 3))
                    np.testing.assert_allclose(labels.numpy(), expected if target == "direct"
                                               else expected - baseline)
                    np.testing.assert_allclose(metadata["baseline"].numpy(), baseline)
                    np.testing.assert_allclose(metadata["reference"].numpy(), expected)

    def test_bound_controls_tail_size_and_small_masks_have_one_pixel(self):
        with patch.object(TemperatureDataset, "get_data", return_value=([], [], [], [], [])):
            dataset = TemperatureDataset("unused", "train", "tomato", "fcdensenet",
                                         "thermal", "none", "nearest", "direct",
                                         reg_output="cwsi", bound=25.)
        thermal = np.arange(1, 9, dtype=np.float32)
        np.testing.assert_allclose(dataset._temperature_targets(thermal, np.ones(8)),
                                   [4.5, 7.5, 1.5])
        dataset.bound = 5.
        np.testing.assert_allclose(dataset._temperature_targets(thermal, np.ones(8)),
                                   [4.5, 8., 1.])

    def test_invalid_options_are_rejected_before_loading(self):
        with patch.object(TemperatureDataset, "get_data") as load:
            for options in ({"reg_output": "other"}, {"bound": 0}, {"bound": 51},
                            {"bound": float("nan")}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    TemperatureDataset("unused", "train", "tomato", "fcdensenet",
                                       "thermal", "none", "nearest", "direct", **options)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
