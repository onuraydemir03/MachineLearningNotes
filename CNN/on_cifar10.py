import torch.optim

from CNN.resnet18 import Resnet18
from Datasets.cifar10 import init_dataloaders
from train_eval_network import train, eval


def train_eval_resnet18(train_dataloader, val_dataloader, test_dataloader, num_epochs, device):
    model = Resnet18(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    train(model, optimizer, num_epochs, train_dataloader, val_dataloader, device, flat=False)
    eval(model, test_dataloader, device, flat=False)


if __name__ == '__main__':
    train_dataloader, val_dataloader, test_dataloader = init_dataloaders(batch_size=128)
    num_epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_eval_resnet18(train_dataloader, val_dataloader, test_dataloader, num_epochs, device)

