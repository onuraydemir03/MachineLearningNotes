import torch.optim
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from CNN.resnet18 import Resnet18
from Datasets.cifar10 import init_dataloaders
from lightning_model import LightningModel
from train_eval_network import train, eval
import lightning as L

from utils import LOG_DIR


def train_eval_resnet18(train_dataloader, val_dataloader, test_dataloader, num_epochs, device):
    model = Resnet18(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    train(model, optimizer, num_epochs, train_dataloader, val_dataloader, device)
    eval(model, test_dataloader, device, flat=False)


def train_eval_resnet18_lightning(train_dataloader, val_dataloader, test_dataloader, num_epochs):
    model = Resnet18(num_classes=10)
    lightning_model = LightningModel(model=model, lr=0.05, num_epochs=num_epochs)
    callbacks = [ModelCheckpoint(save_top_k=1, monitor="val_acc", mode="max", save_last=True)]
    trainer = L.Trainer(max_epochs=num_epochs,
                        accelerator="auto",
                        devices="auto",
                        deterministic=True,
                        callbacks=callbacks,
                        logger=CSVLogger(save_dir=LOG_DIR, name="resnet18_cifar10"))
    trainer.fit(model=lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=lightning_model, dataloaders=test_dataloader)


def load_lightning_model_and_test(test_dataloader, device):
    model = Resnet18(num_classes=10)
    lightning_model = LightningModel.load_from_checkpoint(model=model,
                                                          checkpoint_path="../Logs/resnet18_cifar10/version_0/checkpoints/last.ckpt",
                                                          map_location=device)
    test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    for idx, (features, labels) in enumerate(test_dataloader):
        with torch.inference_mode():
            features = features.to(device)
            labels = labels.to(device)
            logits = lightning_model(features)
            predicted_labels = torch.argmax(logits, dim=1)
            test_acc(labels.cpu(), predicted_labels.cpu())

    print(f"Test accuracy: {test_acc.compute():.2f}")


if __name__ == '__main__':
    train_dataloader, val_dataloader, test_dataloader = init_dataloaders(batch_size=128)
    num_epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_eval_resnet18(train_dataloader, val_dataloader, test_dataloader, num_epochs, device)
    train_eval_resnet18_lightning(train_dataloader, val_dataloader, test_dataloader, num_epochs)
    load_lightning_model_and_test(test_dataloader, device)
