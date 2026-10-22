"""Shared normalization fitted only on training inputs."""

import torch
from torch import nn


class TemperatureNormalizedRegressor(nn.Module):
    def __init__(self, image="thermal", normalize="temp", in_channels=3):
        super().__init__()
        if image not in {"thermal", "visual"}:
            raise ValueError("image must be thermal or visual")
        if normalize not in {"none", "temp", "all"}:
            raise ValueError("normalize must be none, temp, or all")
        self.image = image
        # Controlled by train.normalize.
        self.normalize = normalize
        if normalize != "none":
            channels = in_channels if normalize == "all" else 1
            self.register_buffer("input_mean", torch.zeros(1, channels, 1, 1))
            self.register_buffer("input_std", torch.ones(1, channels, 1, 1))
            self.register_buffer("normalization_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit_normalization(self, train_data):
        """Fit each selected channel across training samples and spatial pixels."""
        if self.normalize == "none" or (self.normalize == "temp" and self.image != "thermal"):
            return
        values = train_data[:, :1] if self.normalize == "temp" else train_data
        values = values.double()
        if values.numel() == 0 or not torch.isfinite(values).all():
            raise ValueError("Training inputs must be nonempty and finite")
        if values.shape[1] != self.input_mean.shape[1]:
            raise ValueError("Training channel count does not match the model")
        self.input_mean.copy_(values.mean(dim=(0, 2, 3), keepdim=True))
        self.input_std.copy_(values.std(dim=(0, 2, 3), unbiased=False, keepdim=True).clamp_min(1e-6))
        self.normalization_fitted.fill_(True)

    def normalize_inputs(self, x):
        if self.normalize == "none" or (self.normalize == "temp" and self.image != "thermal"):
            return x
        if not self.normalization_fitted:
            raise RuntimeError("Fit normalization on training data before prediction")
        if self.normalize == "all":
            if x.shape[1] != self.input_mean.shape[1]:
                raise ValueError("Input channel count does not match the model")
            return (x - self.input_mean) / self.input_std
        temperature = (x[:, :1] - self.input_mean) / self.input_std
        return torch.cat((temperature, x[:, 1:]), dim=1)
