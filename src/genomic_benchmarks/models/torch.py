import os
import sys
from pathlib import Path

import numpy as np

import torch
from torch import nn


# A simple CNN model
class CNN(nn.Module):
    def __init__(self, number_of_classes, vocab_size, embedding_dim, input_len):
        super(CNN, self).__init__()
        if number_of_classes == 2:
            number_of_output_neurons = 1
            loss = torch.nn.functional.binary_cross_entropy_with_logits
            output_activation = nn.Sigmoid()
        else:
            raise Exception("Not implemented for number_of_classes!=2")
            # number_of_output_neurons = number_of_classes
            # loss = torch.nn.CrossEntropyLoss()
            # output_activation = nn.Softmax(dim=)

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
        x = self.output_activation(x)
        return x

    def train_loop(self, dataloader, valid_dataloader, optimizer, experiment):
        for x, y in dataloader:
            optimizer.zero_grad()
            pred = self(x)
            loss = self.loss(pred, y)
            loss.backward()
            optimizer.step()

        #       train acc
        # todo: optimize counting of acc
        size = dataloader.dataset.__len__()
        num_batches = len(dataloader)
        train_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = self(X)
                train_loss += self.loss(pred, y).item()
                correct += (torch.round(pred) == y).sum().item()

        train_loss /= num_batches
        correct /= size

        valid_size = valid_dataloader.dataset.__len__()
        valid_num_batches = len(valid_dataloader)
        valid_loss, valid_correct = 0, 0
        with torch.no_grad():
            for X, y in valid_dataloader:
                pred = self(X)
                valid_loss += self.loss(pred, y).item()
                valid_correct += (torch.round(pred) == y).sum().item()

        valid_loss /= valid_num_batches
        valid_correct /= valid_size

        experiment.log_metric("train_loss", train_loss)
        experiment.log_metric("valid_loss", valid_loss)
        experiment.log_metric("valid_acc", valid_correct)

        print(f"Valid metrics: \n Accuracy: {(100*valid_correct):>0.1f}%, Avg loss: {valid_loss:>8f} \n")
        print(f"Train metrics: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
        return valid_loss

    def train(self, dataloader, valid_loader, epochs, patience, checkpoint_name, experiment):
        optimizer = torch.optim.Adam(self.parameters())
        best_valid_so_far = sys.float_info.max
        epochs_not_improving = 0
        # TODO set: patience, model_path
        # patience = 5
        best_epoch = 0
        checkpoint_name = checkpoint_name + "_checkpoint.pt"
        model_path = Path.home() / "genomic_benchmarks" / "checkpoints" / checkpoint_name
        print(best_valid_so_far)
        for t in range(epochs):
            print(f"Epoch {t}")
            valid_loss = self.train_loop(dataloader, valid_loader, optimizer, experiment)
            if valid_loss <= best_valid_so_far:
                best_valid_so_far = valid_loss
                epochs_not_improving = 0
                best_epoch = t
                print("new best valid loss in epoch " + str(best_epoch) + " -> saving new checkpoint")
                # save model
                torch.save(self.state_dict(), model_path)
            else:
                epochs_not_improving += 1
                print("not improved for epochs:", epochs_not_improving)
                if epochs_not_improving >= patience:
                    # load model
                    self.load_state_dict(torch.load(model_path))
                    print("ending training")
                    print("loading best model from epoch " + str(best_epoch))
                    break

    # TODO: update for multiclass classification datasets
    def test(self, dataloader, positive_label=1, experiment=None):
        size = dataloader.dataset.__len__()
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        tp, p, fp = 0, 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = self(X)
                test_loss += self.loss(pred, y).item()
                correct += (torch.round(pred) == y).sum().item()
                p += (y == positive_label).sum().item()
                if positive_label == 1:
                    tp += (y * pred).sum(dim=0).item()
                    fp += ((1 - y) * pred).sum(dim=0).item()
                else:
                    tp += ((1 - y) * (1 - pred)).sum(dim=0).item()
                    fp += (y * (1 - pred)).sum(dim=0).item()

        print("p ", p, "; tp ", tp, "; fp ", fp)
        recall = tp / p
        precision = tp / (tp + fp)
        print("recall ", recall, "; precision ", precision)
        f1_score = 2 * precision * recall / (precision + recall)

        print("num_batches", num_batches)
        print("correct", correct)
        print("size", size)

        test_loss /= num_batches
        accuracy = correct / size
        print(f"Test metrics: \n Accuracy: {accuracy:>6f}, F1 score: {f1_score:>6f}, Avg loss: {test_loss:>6f} \n")

        experiment.log_metric("test_F1_score", f1_score)
        experiment.log_metric("test_acc", accuracy)
        experiment.log_metric("test_avg_loss", test_loss)
        experiment.log_metric("recall", recall)
        experiment.log_metric("precision", precision)

        return accuracy, f1_score
