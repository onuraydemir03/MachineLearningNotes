import torch
from torch import nn


class Resnet18(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V1")

        # Train only the last fc layer
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        return self.model(x)