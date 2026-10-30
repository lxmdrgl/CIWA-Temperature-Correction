"""Shared thermal regression setup for ResNet backbones."""

import torch
from torch import nn
from models.normalization import TemperatureNormalizedRegressor


class ResNetRegressor(TemperatureNormalizedRegressor):
    def __init__(self, builder, weights, in_channels=3, image="thermal", pretrained=True,
                 freeze_backbone=False, normalize="temp", freeze_conv1=False,
                 mask_init="zero", num_outputs=1, small_stem=False,
                 freeze_layer1=None, freeze_layer4=None):
        super().__init__(image=image, normalize=normalize, in_channels=in_channels,
                         rgb_normalization="imagenet")
        if image == "visual" and in_channels != 3:
            raise ValueError("Visual ResNet inputs must have three RGB channels")
        if image == "thermal" and in_channels not in {1, 2, 3}:
            raise ValueError("Thermal inputs need temperature and up to two mask channels")
        if freeze_backbone and not pretrained:
            raise ValueError("Freezing a backbone requires pretrained weights")
        if mask_init not in {"zero", "default"}:
            raise ValueError("mask_init must be 'zero' or 'default'")

        self.freeze_backbone = freeze_backbone
        self.freeze_conv1 = freeze_conv1
        self.freeze_layer1 = freeze_backbone if freeze_layer1 is None else freeze_layer1
        self.freeze_layer4 = freeze_backbone if freeze_layer4 is None else freeze_layer4
        self.backbone = builder(weights=weights if pretrained else None)
        if small_stem:
            original = self.backbone.conv1
            conv = nn.Conv2d(original.in_channels, original.out_channels,
                             kernel_size=3, stride=1, padding=1, bias=False)
            if pretrained:
                with torch.no_grad():
                    # Retain the center of the pretrained 7x7 RGB filters.
                    conv.weight.copy_(original.weight[:, :, 2:5, 2:5])
            self.backbone.conv1 = conv
            self.backbone.maxpool = nn.Identity()
        if image in {"thermal", "both"}:
            original = self.backbone.conv1
            conv = nn.Conv2d(in_channels, original.out_channels,
                             kernel_size=original.kernel_size, stride=original.stride,
                             padding=original.padding, bias=False)
            if pretrained:
                with torch.no_grad():
                    if image == "both":
                        # Start T from the RGB mean, then scale all four image
                        # channels by 3/4 to preserve the original channel sum.
                        conv.weight[:, :3].copy_(original.weight * (3.0 / 4.0))
                        conv.weight[:, 3:4].copy_(original.weight.mean(dim=1, keepdim=True)
                                                 * (3.0 / 4.0))
                    else:
                        # Preserve the response to the thermal plane repeated across RGB.
                        conv.weight[:, :1].copy_(original.weight.sum(dim=1, keepdim=True))
            if mask_init == "zero":
                with torch.no_grad():
                    mask_start = 4 if image == "both" else 1
                    conv.weight[:, mask_start:].zero_()
            self.backbone.conv1 = conv

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_outputs)
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.fc.requires_grad_(True)
        self.backbone.conv1.requires_grad_(not freeze_conv1)
        self.backbone.layer1.requires_grad_(not self.freeze_layer1)
        self.backbone.layer4.requires_grad_(not self.freeze_layer4)
        if freeze_backbone:
            # Block overrides train convolutions while keeping BatchNorm frozen.
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.requires_grad_(False)
        self.train(self.training)

    def optimizer_parameter_groups(self, lr_conv1, lr_layer=1e-5):
        """Give the stem and selected residual blocks their own learning rates."""
        separate = ("backbone.conv1.", "backbone.layer1.", "backbone.layer4.")
        groups = [{"params": [parameter for name, parameter in self.named_parameters()
                              if parameter.requires_grad and not name.startswith(separate)]}]
        if not self.freeze_conv1:
            groups.append({"params": list(self.backbone.conv1.parameters()),
                           "lr": lr_conv1})
        layers = [parameter for layer in (self.backbone.layer1, self.backbone.layer4)
                  for parameter in layer.parameters() if parameter.requires_grad]
        if layers:
            groups.append({"params": layers, "lr": lr_layer})
        return groups

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            # Freeze BatchNorm statistics as well as the backbone's parameters.
            self.backbone.eval()
            self.backbone.fc.train(mode)
            if not self.freeze_conv1:
                self.backbone.conv1.train(mode)
        self.backbone.layer1.train(mode and not self.freeze_layer1)
        self.backbone.layer4.train(mode and not self.freeze_layer4)
        if self.freeze_backbone:
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, x):
        x = self.normalize_inputs(x)
        return self.backbone(x)
