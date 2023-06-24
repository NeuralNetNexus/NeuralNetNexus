import torch
import torch.nn as nn

class SqueezeNet(nn.Module):
   def __init__(
        self,
        num_classes,
        num_channels,
    ):
    
        super().__init__()

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)


   def forward(self, X):
        return self.model(X)