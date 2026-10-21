"""ImageNet-initialized ResNet-18 for temperature regression."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from models.normalization import TemperatureNormalizedRegressor


class ResNet18Regressor(TemperatureNormalizedRegressor):
    def __init__(self, in_channels=3, image="thermal", pretrained=True,
                 freeze_backbone=False):
        super().__init__(image=image)
        if image == "visual" and in_channels != 3:
            raise ValueError("Visual ResNet inputs must have three BGR channels")
        if image == "thermal" and in_channels not in {1, 2, 3}:
            raise ValueError("Thermal inputs need temperature and up to two mask channels")
        if freeze_backbone and not pretrained:
            raise ValueError("Freezing a backbone requires pretrained weights")

        self.freeze_backbone = freeze_backbone
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
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
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.fc.requires_grad_(True)
            if image == "thermal":
                self.backbone.conv1.requires_grad_(True)
            self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            # Freeze BatchNorm statistics as well as the backbone's parameters.
            self.backbone.eval()
            self.backbone.fc.train(mode)
            if self.image == "thermal":
                self.backbone.conv1.train(mode)
        return self

    def forward(self, x):
        x = self.normalize_inputs(x)
        if self.image == "visual":
            # The existing data loader reads OpenCV BGR images on a 0–255 scale.
            x = x[:, [2, 1, 0]] / 255.
            x = (x - self.rgb_mean) / self.rgb_std
        return self.backbone(x)
