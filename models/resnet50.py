"""ResNet-50 for temperature regression."""

from torchvision.models import ResNet50_Weights, resnet50
from models.resnet import ResNetRegressor


class ResNet50Regressor(ResNetRegressor):
    def __init__(self, in_channels=3, image="thermal", pretrained=True,
                 freeze_backbone=False, normalize="temp", freeze_conv1=False):
        super().__init__(resnet50, ResNet50_Weights.DEFAULT, in_channels=in_channels,
                         image=image, pretrained=pretrained,
                         freeze_backbone=freeze_backbone, normalize=normalize,
                         freeze_conv1=freeze_conv1)
