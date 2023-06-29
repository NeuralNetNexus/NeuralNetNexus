import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.model = _CNN(num_classes)

    def forward(self, X):
        return self.model(X)

class _CNN(nn.Module):
    def __init__(self, num_classes):
        super(_CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 5) # Convolutional layer 1
        self.pool = nn.MaxPool2d(2, 2) # Pooling layer
        self.conv2 = nn.Conv2d(6, 16, 5) # Convolutional layer 2
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
         # Pass data through conv1
        x = self.pool(F.relu(self.conv1(x)))
        # Pass data through conv2
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the data
        x = x.view(-1, 16 * 5 * 5)
        # Pass data through fc1
        x = F.relu(self.fc1(x))
        # Pass data through fc2
        x = F.relu(self.fc2(x))
        # Pass data through fc3
        x = self.fc3(x)
        return x