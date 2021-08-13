import torch
import torch.nn as nn


class TrajModNetwork(nn.Module):
    def __init__(self, numCoord):
        super().__init__()
        self.hidden1 = nn.Linear(1, 128)
        self.relu = nn.ReLU()
        self.hidden2 = nn.Linear(128, 64)
        self.hidden3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, numCoord * 3)
        self.layers = nn.Sequential(self.hidden1, self.relu,
                                    self.hidden2, self.relu,
                                    self.hidden3, self.relu,
                                    self.output)

    def forward(self, x):
        return self.layers(x)


class AngleTrajNet(nn.Module):
    def __init__(self, numAngle):
        super().__init__()
        self.relu = nn.ReLU()
        self.hidden1 = nn.Linear(1, 32)
        self.hidden2 = nn.Linear(32, 64)
        self.hidden3 = nn.Linear(64, 128)
        self.hidden4 = nn.Linear(128, 64)
        self.hidden5 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 2 * numAngle)
        self.layers = nn.Sequential(self.hidden1, self.relu,
                                    self.hidden2, self.relu,
                                    self.hidden3, self.relu,
                                    self.hidden4, self.relu,
                                    self.hidden5, self.relu,
                                    self.output)

    def forward(self, x):
        return self.layers(x)