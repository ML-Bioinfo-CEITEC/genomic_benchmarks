import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tools.dset_getters import dummy_dset, cvsi_dset


# a dummy model with an adjustable number of classes
class NeuralNetwork(nn.Module):
    def __init__(self, number_of_classes):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(200,512)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(512, number_of_classes)
        self.softmax = nn.Softmax(dim=0)
        self.loss = torch.nn.functional.binary_cross_entropy_with_logits

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.softmax(x)

        return x
        
    def train_loop(self, dataloader, optimizer):
        for x, y in dataloader:
            pred = self(x)
            loss = self.loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    def train(self, dataloader, epochs):
        learning_rate = 1e-3
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        for t in range(epochs):
            print(f"Epoch {t}")
            self.train_loop(loader, optimizer)

    def test(self, dataloader):
        size = dataloader.dataset.__len__()
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = self(X)
                test_loss += self.loss(pred, y).item()
                correct += (torch.round(pred) == y).sum().item()

        print('test_loss ', test_loss)
        print('num_batches', num_batches)
        print('correct', correct)
        print('size', size)

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss:{test_loss:>8f} \n")