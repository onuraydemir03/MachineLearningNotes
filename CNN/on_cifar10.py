import torch.optim
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from CNN.cnn_torch_model import CNN
from CNN.resnet18 import Resnet18
from Datasets.cifar10 import init_dataloaders
from lightning_model import LightningModel
from train_eval_network import train, eval
import lightning as L

from utils import LOG_DIR


def train_eval(model, train_dataloader, val_dataloader, test_dataloader, num_epochs, device):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    train(model, optimizer, num_epochs, train_dataloader, val_dataloader, device)
    eval(model, test_dataloader, device, flat=False)


def train_eval_lightning(model, train_dataloader, val_dataloader, test_dataloader, num_epochs):
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


def load_lightning_model_and_test(model, test_dataloader, device):
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
    resnet18 = Resnet18(num_classes=10)
    custom_cnn = CNN(in_channels=3, num_classes=10)
    train_dataloader, val_dataloader, test_dataloader = init_dataloaders(batch_size=128)
    num_epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # train_eval(resnet18, train_dataloader, val_dataloader, test_dataloader, num_epochs, device)
    # train_eval_lightning(resnet18, train_dataloader, val_dataloader, test_dataloader, num_epochs)
    # load_lightning_model_and_test(resnet18, test_dataloader, device)

    train_eval(custom_cnn, train_dataloader, val_dataloader, test_dataloader, num_epochs, device)
    train_eval_lightning(custom_cnn, train_dataloader, val_dataloader, test_dataloader, num_epochs)
    load_lightning_model_and_test(custom_cnn, test_dataloader, device)
