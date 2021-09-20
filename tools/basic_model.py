import os
import torch
from torch import nn
from torch.utils.data import DataLoader

# a dummy model with an adjustable number of classes
class NeuralNetwork(nn.Module):
    def __init__(self, number_of_classes):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(100,512)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(512, number_of_classes)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.softmax(x)

        return x
    
