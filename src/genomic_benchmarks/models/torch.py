import os

import numpy as np

import torch
from torch import nn
from sklearn import metrics


# A simple CNN model
class CNN(nn.Module):
    def __init__(self, number_of_classes, vocab_size, embedding_dim, input_len):
        super(CNN, self).__init__()
        if number_of_classes == 2:
            self.is_multiclass = False
            number_of_output_neurons = 1
            output_activation = nn.Sigmoid()
            loss = torch.nn.functional.binary_cross_entropy_with_logits
        elif number_of_classes > 2:
            self.is_multiclass = True
            print("number_of_classes > 2")
            number_of_output_neurons = number_of_classes
            output_activation = None
            loss = torch.nn.functional.cross_entropy
        else:
            raise Exception("number_of_classes < 2 is not a correct input")

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=16, kernel_size=8, bias=True)
        self.norm1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=8, bias=True)
        self.norm2 = nn.BatchNorm1d(8)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=8, bias=True)
        self.norm3 = nn.BatchNorm1d(4)
        self.pool3 = nn.MaxPool1d(2)

        #         compute output shape of conv layers
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(self.count_flatten_size(input_len), 512)
        self.lin2 = nn.Linear(512, number_of_output_neurons)
        self.output_activation = output_activation
        self.loss = loss

    def count_flatten_size(self, input_len):
        zeros = torch.zeros([1, input_len], dtype=torch.long)
        x = self.embeddings(zeros)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.flatten(x)
        return x.size()[1]

    def forward(self, x):
        x = self.embeddings(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.lin1(x)
        x = self.lin2(x)
        if self.output_activation != None:
            x = self.output_activation(x)
        return x

    def train_loop(self, dataloader, optimizer):
        for x, y in dataloader:
            optimizer.zero_grad()
            pred = self(x)
            if (self.is_multiclass):
                y = y[:,0].long()
            loss = self.loss(pred, y)
            loss.backward()
            optimizer.step()

        #       train acc
        # TODO: optimize counting of acc
        size = dataloader.dataset.__len__()
        num_batches = len(dataloader)
        train_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = self(X)
                if (self.is_multiclass):
                    y = y[:,0].long()
                    correct += (torch.argmax(pred) == y).sum().item()
                else:
                    correct += (torch.round(pred) == y).sum().item()
                train_loss += self.loss(pred, y).item()

        train_loss /= num_batches
        correct /= size
        print(f"Train metrics: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")


    def train(self, dataloader, epochs):
        optimizer = torch.optim.Adam(self.parameters())
        for t in range(epochs):
            print(f"Epoch {t}")
            self.train_loop(dataloader, optimizer)


    def test(self, dataloader, positive_label = 1):
        size = dataloader.dataset.__len__()
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        tp, p, fp = 0, 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = self(X)
                correct += (torch.round(pred) == y).sum().item()
                test_loss += self.loss(pred, y).item()
                p += (y == positive_label).sum().item() 
                if(positive_label == 1):
                    tp += (y * pred).sum(dim=0).item()
                    fp += ((1 - y) * pred).sum(dim=0).item()
                else:
                    tp += ((1 - y) * (1 - pred)).sum(dim=0).item()
                    fp += (y * (1 - pred)).sum(dim=0).item()

        print("p ", p, "; tp ", tp, "; fp ", fp)
        recall = tp / p
        precision = tp / (tp + fp)
        print("recall = (tp / p) = ", recall, "; precision = (tp / (tp + fp)) = ", precision)
        f1_score = 2 * precision * recall / (precision + recall)
        
        print("f1_score = 2 * precision * recall / (precision + recall) =", f1_score)
        print("num_batches ", num_batches)
        print("correct ", correct)
        print("size ", size)

        test_loss /= num_batches
        accuracy = correct / size
        print(f"Test metrics: \n Accuracy: {accuracy:>6f}, F1 score: {f1_score:>6f}, Avg loss: {test_loss:>6f} \n")
        
        return accuracy, f1_score


    def test_multiclass(self, dataloader, class_count, positive_label = 1):
        size = dataloader.dataset.__len__()
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        tp, p, fp = [], [], []
        for i in range(class_count):
            p.append(0)
            tp.append(0)
            fp.append(0)
        
        # using confusion matrix sklearn
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for X, y in dataloader:
                y = y[:,0].long()
                pred = self(X)
                arg_max_pred = torch.argmax(pred, dim=1)

                all_predictions.extend(arg_max_pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

                correct += (arg_max_pred == y).sum().item()
                test_loss += self.loss(pred, y).item()


        metrics.confusion_matrix(all_labels, all_predictions)
        print(metrics.classification_report(all_labels, all_predictions, digits=3))
        # HEADER sklearn.metrics.f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        f1_score = metrics.f1_score(all_labels, all_predictions, average=None)
        print(f1_score)

        print("num_batches ", num_batches)
        print("correct ", correct)
        print("size ", size)

        test_loss /= num_batches
        accuracy = correct / size
        print(f"Test metrics: \n Accuracy: {accuracy:>6f}, F1 score: {f1_score:>6f}, Avg loss: {test_loss:>6f} \n")

        return accuracy, f1_score
        # return accuracy
