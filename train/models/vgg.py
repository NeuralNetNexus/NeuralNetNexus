import torch.nn as nn
import torchvision.models as models

class VGG(nn.Module):
    def __init__(
        self,
        num_classes,
        num_channels,
    ):
        super().__init__()

        self.model = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)

        if num_channels != 3:
            self.model.features[0] = nn.Conv2d(num_channels, 64, kernel_size=(11,11), stride=(4,4))
        else:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        self.model.classifier[6].requires_grad = True

    def forward(self, X):
        return self.model(X)