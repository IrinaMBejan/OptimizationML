"""File contains all the architectures we used during experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBatch(nn.Module):
    """
    Architecture that has 4 convolutional layers
    """

    def __init__(self, input_channels, size):
        super(SimpleBatch, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3),  # 30*30
            nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3),  # 28*28
            nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 14*14
            nn.Conv2d(32, 64, kernel_size=3),  # 12*12
            nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3),  # 10*10
            nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            nn.MaxPool2d(2)  # 5*5
        )

        self.size_ = ((size - 4) // 2 - 4) // 2
        self.linears = nn.Sequential(
            nn.Linear(128 * (self.size_ ** 2), 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        # to keep track of the training information
        self.val_losses = []
        self.train_losses = []
        self.val_accuracies = []
        self.train_accuracies = []

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(-1, (self.size_ ** 2) * 128)
        x = self.linears(x)
        out = F.log_softmax(x, dim=0)
        return out


class MiddleBatch(nn.Module):
    """
    Architecture that has 6 convolutional layers
    """

    def __init__(self, input_channels, size):
        super(MiddleBatch, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),  # 30*30
            nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 28*28
            nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 16*16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 11*11
            nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 10*10
            nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 11*11
            nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 10*10
            nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )

        self.size_ = ((size // 2) // 2) // 2

        self.linears = nn.Sequential(
            nn.Linear(512 * (self.size_ ** 2), 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

        # to keep track of the training information
        self.val_losses = []
        self.train_losses = []
        self.val_accuracies = []
        self.train_accuracies = []

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(-1, (self.size_ ** 2) * 512)
        x = self.linears(x)

        out = F.log_softmax(x, dim=0)
        return out


class ComplexBatch(nn.Module):
    """
    Architecture that has 9 convolutional layers
    """

    def __init__(self, input_channels, size):
        super(ComplexBatch, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),  # 30*30
            nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 28*28
            nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 28*28
            nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 16*16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 11*11
            nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 10*10
            nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 10*10
            nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 11*11
            nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 10*10
            nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 10*10
            nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )

        self.size_ = ((size // 2) // 2) // 2

        self.linears = nn.Sequential(
            nn.Linear(512 * (self.size_ ** 2), 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

        # to keep track of the training information
        self.val_losses = []
        self.train_losses = []
        self.val_accuracies = []
        self.train_accuracies = []

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(-1, (self.size_ ** 2) * 512)
        x = self.linears(x)

        out = F.log_softmax(x, dim=0)
        return out
