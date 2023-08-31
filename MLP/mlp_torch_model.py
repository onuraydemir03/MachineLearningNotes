from torch import nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_classes: int):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.fc = nn.Sequential(nn.Linear(self.num_features, 50),
                                nn.ReLU(),

                                nn.Linear(50, 25),
                                nn.ReLU(),

                                nn.Linear(25, self.num_classes))

    def forward(self, x):
        logits = self.fc(x)
        return logits



