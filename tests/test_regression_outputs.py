import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

from evaluate import calculate_metrics, evaluate_checkpoint, write_metrics
from main import main
from models.cnn import CNNRegressor
from models.mlp import MLPRegressor
from models.resnet18 import ResNet18Regressor
from models.resnet34 import ResNet34Regressor
from models.resnet50 import ResNet50Regressor
from utils.data_pipeline import create_loader
from utils.training import evaluate_model, train_model
from visualize import main as visualize


class RegressionOutputTests(unittest.TestCase):
    def config(self):
        cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/config.yaml")
        cfg.dataset.image = "thermal"
        cfg.dataset.reg_output = "cwsi"
        cfg.train.normalize = "none"
        cfg.train.batch_size = 2
        cfg.train.epochs = 1
        cfg.train.model = "mlp"
        return cfg

    def test_every_architecture_restores_three_output_checkpoints(self):
        cfg = self.config()
        data = torch.zeros(2, 1, 32, 32)
        baseline = torch.tensor([[10., 30., 5.], [20., 40., 15.]])
        delta = torch.tensor([2., 4., 6.])
        reference = baseline + delta
        builders = {
            "mlp": lambda: MLPRegressor(1024, in_channels=1, num_outputs=3, normalize="none"),
            "cnn": lambda: CNNRegressor(in_channels=1, num_outputs=3, normalize="none"),
            "resnet18": lambda: ResNet18Regressor(in_channels=1, num_outputs=3,
                                                 normalize="none", pretrained=False),
            "resnet34": lambda: ResNet34Regressor(in_channels=1, num_outputs=3,
                                                 normalize="none", pretrained=False),
            "resnet50": lambda: ResNet50Regressor(in_channels=1, num_outputs=3,
                                                 normalize="none", pretrained=False,
                                                 crop=cfg.dataset.crop),
        }
        with TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            for name, build in builders.items():
                model = build()
                model.eval()
                with torch.no_grad():
                    model(data)  # Initialize the CNN's lazy layer before saving.
                    head = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)][-1]
                    head.weight.zero_()
                    head.bias.copy_(delta)
                cfg.train.model = name
                for target in ("direct", "residual"):
                    cfg.dataset.target = target
                    checkpoint = Path(directory) / f"{name}_{target}.pt"
                    torch.save({"config": OmegaConf.to_container(cfg), "epoch": 1,
                                "model_state_dict": model.state_dict()}, checkpoint)
                    for baseline_only in (False, True):
                        with self.subTest(model=name, target=target, baseline=baseline_only), patch(
                            "evaluate.load_dataset", return_value=(data, reference,
                            {"baseline": baseline, "reference": reference})
                        ):
                            rows = evaluate_checkpoint(checkpoint,
                                include_baseline=baseline_only,
                                output=Path(directory) / "metrics.csv")
                            self.assertEqual([row["output"] for row in rows],
                                             ["canopy", "upper_tail", "lower_tail"])
                            expected = (delta.numpy() if baseline_only else
                                        np.zeros(3) if target == "residual" else
                                        baseline.numpy().mean(axis=0))
                            np.testing.assert_allclose([row["mae"] for row in rows], expected)
                            self.assertTrue(all(row["samples"] == 2 for row in rows))

    def test_training_saves_and_restores_both_output_modes(self):
        cfg = self.config()
        data = torch.rand(2, 1, 4, 4)
        with TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            cfg.train.output = directory
            for mode, width in (("canopy", 1), ("cwsi", 3)):
                cfg.dataset.reg_output = mode
                labels = torch.ones(2, width)
                with patch("main.load_dataset", return_value=(data, labels)):
                    main.__wrapped__(cfg)
                checkpoint = next(Path(directory).glob(f"*_reg-{mode}*.pt"))
                saved = torch.load(checkpoint, weights_only=True)
                head = saved["model_state_dict"]["linear_relu_stack.4.weight"]
                self.assertEqual(head.shape[0], width)
                self.assertEqual(saved["config"]["dataset"]["reg_output"], mode)

    def test_training_rejects_broadcasting_between_output_modes(self):
        model = MLPRegressor(16, in_channels=1, normalize="none")
        loader = create_loader(torch.zeros(2, 1, 4, 4), torch.zeros(2, 3), 2, False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with self.assertRaisesRegex(ValueError, "output shape"):
            train_model(model, loader, torch.nn.MSELoss(), optimizer, "cpu")
        with self.assertRaisesRegex(ValueError, "output shape"):
            evaluate_model(model, loader, torch.nn.MSELoss(), "cpu")

    def test_metrics_reject_flattened_or_incompatible_cwsi_outputs(self):
        cfg = self.config()
        with TemporaryDirectory() as directory:
            for reference, prediction in (([1, 2, 3], [1, 2, 3]),
                                          ([[1, 2, 3]], [[1]])):
                with self.assertRaisesRegex(ValueError, "shape"):
                    write_metrics(cfg, "test", reference, [("model", prediction)],
                                  Path(directory) / "metrics.csv")

    def test_visualization_displays_each_temperature_output(self):
        cfg = self.config()
        cfg.visualize.index = 0
        image = np.ones((4, 4), dtype=np.float32)
        source = ([image], [np.zeros((4, 4, 3), dtype=np.uint8)],
                  [image], [image], [image])
        with TemporaryDirectory() as directory:
            cfg.visualize.output = directory
            for mode, image_type in (("canopy", "thermal"), ("cwsi", "thermal"),
                                     ("canopy", "both"), ("cwsi", "both")):
                cfg.dataset.reg_output = mode
                cfg.dataset.image = image_type
                with patch("utils.dataset.TemperatureDataset.get_data", return_value=source):
                    visualize.__wrapped__(cfg)
                paths = [path for path in Path(directory).rglob("*.png")
                         if mode in path.name and f"_{image_type}_" in path.name]
                self.assertEqual(len(paths), 1)

    def test_cwsi_visualization_metrics_use_all_three_temperatures(self):
        cfg = self.config()
        actual = np.array([20., 35., 10.])
        baseline = np.array([22., 31., 11.])
        dataset = MagicMock()
        dataset.__len__.return_value = 1
        dataset.get_sample.return_value = (np.zeros((4, 4, 3)), actual,
            {"baseline": baseline, "thermal_label_mask": np.ones((4, 4))})
        dataset.visual_images = [np.zeros((4, 4, 3), dtype=np.uint8)]
        dataset.visual_hard_masks = dataset.visual_soft_masks = dataset.visual_label_masks = [np.ones((4, 4))]
        with TemporaryDirectory() as directory:
            cfg.visualize.output = directory
            with patch("visualize.TemperatureDataset", return_value=dataset), patch(
                "visualize.calculate_metrics", wraps=calculate_metrics
            ) as metrics, patch("visualize.Figure.savefig", autospec=True) as save:
                visualize.__wrapped__(cfg)
            metrics.assert_called_once()
            np.testing.assert_array_equal(metrics.call_args.args[0], actual)
            np.testing.assert_array_equal(metrics.call_args.args[1], baseline)
            figure = save.call_args.args[0]
            details = figure.axes[-1].texts[0].get_text()
            for label in ("Canopy", "Upper bound", "Lower bound"):
                self.assertIn(label, details)
            self.assertIn("all three temperatures", details)
            self.assertIn("MSE: 7.0000", details)
            self.assertIn("MAE: 2.3333", details)


if __name__ == "__main__":
    unittest.main()
