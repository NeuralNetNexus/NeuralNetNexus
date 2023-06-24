import torch
import torch.nn as nn
import torchvision.models as models

class SqueezeNet(nn.Module):
   def __init__(
        self,
        num_classes,
        num_channels,
    ):
    
        super().__init__()

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', weights=models.SqueezeNet1_0_Weights.DEFAULT)


   def forward(self, X):
        return self.model(X)