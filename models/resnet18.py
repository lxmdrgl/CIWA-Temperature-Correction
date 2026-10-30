"""ResNet-18 for temperature regression."""

from torchvision.models import ResNet18_Weights, resnet18
from models.resnet import ResNetRegressor


class ResNet18Regressor(ResNetRegressor):
    def __init__(self, in_channels=3, image="thermal", pretrained=True,
                 freeze_backbone=False, normalize="temp", freeze_conv1=False,
                 mask_init="zero", num_outputs=1, freeze_layer1=None, freeze_layer4=None):
        super().__init__(resnet18, ResNet18_Weights.DEFAULT, in_channels=in_channels,
                         image=image, pretrained=pretrained,
                         freeze_backbone=freeze_backbone, normalize=normalize,
                         freeze_conv1=freeze_conv1, mask_init=mask_init,
                         num_outputs=num_outputs,
                         freeze_layer1=freeze_layer1, freeze_layer4=freeze_layer4)
