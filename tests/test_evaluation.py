import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from evaluate import calculate_metrics
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


if __name__ == "__main__":
    unittest.main()
