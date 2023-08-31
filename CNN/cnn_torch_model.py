from torch import nn


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=5),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=32, out_features=20),
            nn.BatchNorm1d(20),
            nn.ReLU(),

            nn.Linear(in_features=20, out_features=self.num_classes))

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.flatten(start_dim=1)
        logits = self.fc_layers(x)
        return logits