import os

import numpy as np

import torch
from torch import nn
from sklearn import metrics


# A simple CNN model
class CNN(nn.Module):
    def __init__(self, number_of_classes, vocab_size, embedding_dim, input_len, device):
        super(CNN, self).__init__()

        self.device = device

        if number_of_classes == 2:
            self.is_multiclass = False
            number_of_output_neurons = 1
            loss = torch.nn.functional.binary_cross_entropy_with_logits
            output_activation = nn.Sigmoid()
        else:
            self.is_multiclass = True
            number_of_output_neurons = number_of_classes
            loss = torch.nn.CrossEntropyLoss()
            output_activation = lambda x: x

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.cnn_model = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=16, kernel_size=8, bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=8, bias=True),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=8, bias=True),
            nn.BatchNorm1d(4),
            nn.MaxPool1d(2),

            nn.Flatten()
        )
        self.dense_model = nn.Sequential(
            nn.Linear(self.count_flatten_size(input_len), 512),
            nn.Linear(512, number_of_output_neurons)
        )
        self.output_activation = output_activation
        self.loss = loss

    def count_flatten_size(self, input_len):
        zeros = torch.zeros([1, input_len], dtype=torch.long)
        x = self.embeddings(zeros)
        x = x.transpose(1, 2)
        x = self.cnn_model(x)
        return x.size()[1]

    def forward(self, x):
        x = self.embeddings(x)
        x = x.transpose(1, 2)
        x = self.cnn_model(x)
        x = self.dense_model(x)
        x = self.output_activation(x)
        return x

    def train_loop(self, dataloader, optimizer):

        train_loss, correct = 0, 0
        self.train()
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            if self.is_multiclass:
                y = y[:, 0].long()

            # Compute prediction error
            pred = self(x)
            loss = self.loss(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += self.loss(pred, y).item()

            if self.is_multiclass:
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            else:
                correct += (torch.round(pred) == y).sum().item()

        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        train_loss /= num_batches
        correct /= size
        print(f"Train metrics: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")

    def fit(self, dataloader, epochs):
        optimizer = torch.optim.Adam(self.parameters())
        for t in range(epochs):
            print(f"Epoch {t}")
            self.train_loop(dataloader, optimizer)

    def test(self, dataloader):

        num_batches = len(dataloader)
        self.eval()

        all_predictions = []
        all_labels = []
        test_loss = 0

        with torch.no_grad():
            for X, y in dataloader:
                if self.is_multiclass:
                    y = y[:, 0].long()

                output = self(X)
                test_loss += self.loss(output, y).item()

                if self.is_multiclass:
                    pred = output.argmax(1)
                else:
                    pred = torch.round(output)

                all_labels.extend(y.cpu().numpy())
                all_predictions.extend(pred.cpu().numpy())

        average = 'micro' if self.is_multiclass else 'binary'
        acc = metrics.accuracy_score(all_labels, all_predictions)
        f1 = metrics.f1_score(all_labels, all_predictions, average=average, zero_division=0)
        test_loss /= num_batches

        print(
             f"Test metrics: \n Accuracy: {float(acc):>6f}, F1 score: {float(f1):>6f}, Avg loss: {float(test_loss):>6f} \n"
        )

        return acc, f1