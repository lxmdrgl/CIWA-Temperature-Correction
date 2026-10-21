"""Shared training-set temperature normalization for regression models."""

import torch
from torch import nn


class TemperatureNormalizedRegressor(nn.Module):
    def __init__(self, image="thermal", normalize=True):
        super().__init__()
        if image not in {"thermal", "visual"}:
            raise ValueError("image must be thermal or visual")
        self.image = image
        # Disabled when restoring older MLP/CNN checkpoints trained on raw inputs.
        self.normalize = normalize
        if normalize:
            self.register_buffer("thermal_mean", torch.tensor(0.))
            self.register_buffer("thermal_std", torch.tensor(1.))
            self.register_buffer("thermal_normalization_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit_normalization(self, train_data):
        """Fit one temperature mean/std using only pixels in the training split."""
        if self.normalize and self.image == "thermal":
            temperature = train_data[:, 0].double()
            if temperature.numel() == 0 or not torch.isfinite(temperature).all():
                raise ValueError("Training temperatures must be nonempty and finite")
            self.thermal_mean.copy_(temperature.mean())
            self.thermal_std.copy_(temperature.std(unbiased=False).clamp_min(1e-6))
            self.thermal_normalization_fitted.fill_(True)

    def normalize_inputs(self, x):
        if self.normalize and self.image == "thermal":
            if not self.thermal_normalization_fitted:
                raise RuntimeError("Fit thermal normalization on training data before prediction")
            temperature = (x[:, :1] - self.thermal_mean) / self.thermal_std
            return torch.cat((temperature, x[:, 1:]), dim=1)
        return x
