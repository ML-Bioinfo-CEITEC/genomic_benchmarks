import os
import torch
from torch import nn
from torch.utils.data import DataLoader

# a dummy model with an adjustable number of classes
class NeuralNetwork(nn.Module):
    def __init__(self, number_of_classes):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10*100, 512),
            nn.ReLU(),
            nn.Linear(512, number_of_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        pred_probab = nn.Softmax(dim=1)(logits)
        return pred_probab
    
