"""Shared thermal regression setup for ResNet backbones."""

import torch
from torch import nn
from models.normalization import TemperatureNormalizedRegressor


class ResNetRegressor(TemperatureNormalizedRegressor):
    def __init__(self, builder, weights, in_channels=3, image="thermal", pretrained=True,
                 freeze_backbone=False, normalize="temp", freeze_conv1=False):
        super().__init__(image=image, normalize=normalize, in_channels=in_channels)
        if image == "visual" and in_channels != 3:
            raise ValueError("Visual ResNet inputs must have three BGR channels")
        if image == "thermal" and in_channels not in {1, 2, 3}:
            raise ValueError("Thermal inputs need temperature and up to two mask channels")
        if freeze_backbone and not pretrained:
            raise ValueError("Freezing a backbone requires pretrained weights")

        self.freeze_backbone = freeze_backbone
        self.freeze_conv1 = freeze_conv1
        self.backbone = builder(weights=weights if pretrained else None)
        if image == "thermal":
            original = self.backbone.conv1
            conv = nn.Conv2d(in_channels, original.out_channels,
                             kernel_size=original.kernel_size, stride=original.stride,
                             padding=original.padding, bias=False)
            if pretrained:
                with torch.no_grad():
                    # Initially equivalent to feeding the thermal plane into all RGB channels.
                    conv.weight.zero_()
                    conv.weight[:, :1].copy_(original.weight.sum(dim=1, keepdim=True))
            self.backbone.conv1 = conv

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.fc.requires_grad_(True)
        self.backbone.conv1.requires_grad_(not freeze_conv1)
        self.train(self.training)

    def optimizer_parameter_groups(self, learning_rate_conv1):
        """Use a separate learning rate for trainable input-convolution weights."""
        groups = [{"params": [parameter for name, parameter in self.named_parameters()
                              if parameter.requires_grad and not name.startswith("backbone.conv1.")]}]
        if not self.freeze_conv1:
            groups.append({"params": list(self.backbone.conv1.parameters()),
                           "lr": learning_rate_conv1})
        return groups

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            # Freeze BatchNorm statistics as well as the backbone's parameters.
            self.backbone.eval()
            self.backbone.fc.train(mode)
            if not self.freeze_conv1:
                self.backbone.conv1.train(mode)
        return self

    def forward(self, x):
        x = self.normalize_inputs(x)
        return self.backbone(x)
