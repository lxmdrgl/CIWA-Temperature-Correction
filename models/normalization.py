"""Model-specific RGB normalization and training-fitted thermal statistics."""

import torch
from torch import nn


class TemperatureNormalizedRegressor(nn.Module):
    def __init__(self, image="thermal", normalize="temp", in_channels=3,
                 rgb_normalization="training"):
        super().__init__()
        if image not in {"thermal", "visual", "both"}:
            raise ValueError("image must be thermal, visual, or both")
        if image == "both" and in_channels not in {4, 5, 6}:
            raise ValueError("Combined inputs need RGB, temperature, and up to two mask channels")
        if image == "visual" and in_channels != 3:
            raise ValueError("Visual inputs must have three RGB channels")
        if normalize not in {"none", "temp", "all"}:
            raise ValueError("normalize must be none, temp, or all")
        if rgb_normalization not in {"training", "imagenet"}:
            raise ValueError("rgb_normalization must be training or imagenet")
        self.image = image
        self.normalize = normalize
        self.rgb_normalization = rgb_normalization
        self.has_rgb = image in {"visual", "both"}
        self.normalization_start = 3 if self.has_rgb else 0
        self.normalization_end = (in_channels if normalize == "all" else
                                  min(in_channels, self.normalization_start + 1))
        if self.has_rgb:
            mean = [0.485, 0.456, 0.406] if rgb_normalization == "imagenet" else [0., 0., 0.]
            std = [0.229, 0.224, 0.225] if rgb_normalization == "imagenet" else [1., 1., 1.]
            self.register_buffer("rgb_mean", torch.tensor(mean).reshape(1, 3, 1, 1))
            self.register_buffer("rgb_std", torch.tensor(std).reshape(1, 3, 1, 1))
            self.register_buffer("rgb_normalization_fitted", torch.tensor(rgb_normalization == "imagenet"))
        self.normalize_thermal = normalize != "none" and image != "visual"
        if self.normalize_thermal:
            channels = self.normalization_end - self.normalization_start
            self.register_buffer("input_mean", torch.zeros(1, channels, 1, 1))
            self.register_buffer("input_std", torch.ones(1, channels, 1, 1))
            self.register_buffer("normalization_fitted", torch.tensor(False))

    @staticmethod
    def _statistics(values):
        values = values.double()
        if values.numel() == 0 or not torch.isfinite(values).all():
            raise ValueError("Training inputs must be nonempty and finite")
        return (values.mean(dim=(0, 2, 3), keepdim=True),
                values.std(dim=(0, 2, 3), unbiased=False, keepdim=True).clamp_min(1e-6))

    @torch.no_grad()
    def fit_normalization(self, train_data):
        """Fit each RGB/thermal channel across training samples and spatial pixels."""
        if self.has_rgb and self.rgb_normalization == "training":
            mean, std = self._statistics(train_data[:, :3].double() / 255.0)
            self.rgb_mean.copy_(mean)
            self.rgb_std.copy_(std)
            self.rgb_normalization_fitted.fill_(True)
        if self.normalize_thermal:
            values = train_data[:, self.normalization_start:self.normalization_end]
            if values.shape[1] != self.input_mean.shape[1]:
                raise ValueError("Training channel count does not match the model")
            mean, std = self._statistics(values)
            self.input_mean.copy_(mean)
            self.input_std.copy_(std)
            self.normalization_fitted.fill_(True)

    def normalize_inputs(self, x):
        if self.has_rgb:
            if not self.rgb_normalization_fitted:
                raise RuntimeError("Fit normalization on training data before prediction")
            rgb = (x[:, :3] / 255.0 - self.rgb_mean) / self.rgb_std
            x = torch.cat((rgb, x[:, 3:]), dim=1)
        if not self.normalize_thermal:
            return x
        if not self.normalization_fitted:
            raise RuntimeError("Fit normalization on training data before prediction")
        start, end = self.normalization_start, self.normalization_end
        if x[:, start:end].shape[1] != self.input_mean.shape[1]:
            raise ValueError("Input channel count does not match the model")
        values = (x[:, start:end] - self.input_mean) / self.input_std
        return torch.cat((x[:, :start], values, x[:, end:]), dim=1)
