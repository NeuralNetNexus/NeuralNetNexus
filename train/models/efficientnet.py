import torch.nn as nn
import torchvision.models as models

class EfficientNet(nn.Module):
    def __init__(
        self,
        num_classes,
        num_channels,
    ):
        super().__init__()

        self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

        if num_channels != 3:
            self.model.features[0] = nn.Conv2d(num_channels, 24, kernel_size=(3,3), stride=(2,2), padding=(1, 1), bias=False)
        else:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.classifier[1] = nn.Linear(1280, num_classes)
        self.model.classifier[1].requires_grad = True

    def forward(self, X):
        return self.model(X)