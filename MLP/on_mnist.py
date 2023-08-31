import torch.optim

from Datasets.mnist import init_dataloaders
from MLP.mlp_torch_model import MultiLayerPerceptron

from train_eval_network import train, eval


def train_eval_mlp(train_dataloader, val_dataloader, test_dataloader, num_epochs, device):
    model = MultiLayerPerceptron(num_features=784, num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    train(model, optimizer, num_epochs, train_dataloader, val_dataloader, device)
    eval(model, test_dataloader, device)


if __name__ == '__main__':
    train_dataloader, val_dataloader, test_dataloader = init_dataloaders()
    num_epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_eval_mlp(train_dataloader, val_dataloader, test_dataloader, num_epochs, device)

