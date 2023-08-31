import torch
import torchmetrics
from tqdm import tqdm
import torch.nn.functional as F


def train(model, optimizer, num_epochs, train_dataloader, val_dataloader, device, num_classes=10):
    pbar = tqdm(range(num_epochs))
    train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    model.to(device)
    for epoch_no in pbar:
        model.train()
        for idx, (features, labels) in enumerate(train_dataloader):
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            train_loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            pred_labels = torch.argmax(logits, dim=1)
            train_acc(pred_labels.cpu(), labels.cpu())

        model.eval()
        for idx, (features, labels) in enumerate(val_dataloader):
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            val_loss = F.cross_entropy(logits, labels)
            pred_labels = torch.argmax(logits, dim=1)
            val_acc(pred_labels.cpu(), labels.cpu())
        pbar.set_postfix_str(f"Ep: {epoch_no} > train_loss: {train_loss:.2f} | val_acc: {train_acc.compute():.2f} | val_loss: {val_loss:.2f} | val_acc: {val_acc.compute():.2f}")
        print()


def eval(model, test_dataloader, device, num_classes=10, flat=True):
    test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    model.to(device)
    model.eval()
    for idx, (features, labels) in enumerate(test_dataloader):
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        pred_labels = torch.argmax(logits, dim=1)
        test_acc(pred_labels.cpu(), labels.cpu())
    print(f"Test accuracy: {test_acc.compute():.2f}")