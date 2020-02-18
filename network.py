import torch.nn as nn
import torch.nn.functional as F

class SimpleNetwork(nn.Module):
    def __init__(self, id, unit, od):
        super(SimpleNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.L1 = nn.Linear(id, unit)
        self.L2 = nn.Linear(unit, unit)
        self.L3 = nn.Linear(unit, od)
    
    def forward(self, x):
        x = self.L1(x)
        x = self.relu(x)
        x = self.L2(x)
        x = self.relu(x)
        x = self.L3(x)
        return x

class ConvNetwork(nn.Module):
    def __init__(self, id, unit, od):
        super(ConvNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=1)

        self.conv1 = nn.Conv2d(1, id, 2)
        self.conv2 = nn.Conv2d(id, 32, 2)

        self.L1 = nn.Linear(32, unit)
        self.L2 = nn.Linear(unit, unit)
        self.L3 = nn.Linear(unit, od)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(-1, 32)
        x = self.L1(x)
        x = self.relu(x)
        x = self.L2(x)
        x = self.relu(x)
        x = self.L3(x)
        return x