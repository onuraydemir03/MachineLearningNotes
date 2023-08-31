import torch.optim
import torchmetrics
from tqdm import tqdm

from Datasets.mnist import init_dataloaders
from MLP.mlp_torch_model import MultiLayerPerceptron
import torch.nn.functional as F


def train(model, optimizer, num_epochs, train_dataloader, val_dataloader):
    pbar = tqdm(range(num_epochs))
    train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    for epoch_no in pbar:
        model.train()
        for idx, (features, labels) in enumerate(train_dataloader):
            features = features.flatten(start_dim=1)
            logits = model(features)
            train_loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            pred_labels = torch.argmax(logits, dim=1)
            train_acc(pred_labels, labels)

        model.eval()
        for idx, (features, labels) in enumerate(val_dataloader):
            features = features.flatten(start_dim=1)
            logits = model(features)
            val_loss = F.cross_entropy(logits, labels)
            pred_labels = torch.argmax(logits, dim=1)
            val_acc(pred_labels, labels)
        pbar.set_postfix_str(f"Ep: {epoch_no} > train_loss: {train_loss:.2f} | val_acc: {train_acc.compute():.2f} | val_loss: {val_loss:.2f} | val_acc: {val_acc.compute():.2f}")
        print()




if __name__ == '__main__':
    train_dataloader, val_dataloader, test_dataloader = init_dataloaders()
    model = MultiLayerPerceptron(num_features=784, num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    num_epochs = 10

    train(model, optimizer, num_epochs, train_dataloader, val_dataloader)
