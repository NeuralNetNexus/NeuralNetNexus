import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }
    weights = {
        18: models.ResNet18_Weights.DEFAULT,
        34: models.ResNet34_Weights.DEFAULT,
        50: models.ResNet50_Weights.DEFAULT,
        101: models.ResNet101_Weights.DEFAULT,
        152: models.ResNet152_Weights.DEFAULT,
    }

    def __init__(
        self,
        resnet_version,
        num_classes,
        num_channels,
    ):
        super().__init__()

        self.model = self.resnets[resnet_version](weights=self.weights[resnet_version])

        if num_channels != 3:
            self.model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7)
        else:
            for param in self.model.parameters():
                param.requires_grad = False
        
        linear_size = list(self.model.children())[-1].in_features
        self.model.fc = nn.Linear(linear_size, num_classes)
        self.model.fc.requires_grad = True

    def forward(self, X):
        return self.model(X)