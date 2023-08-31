import torch.optim
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from Datasets.mnist import init_dataloaders
from MLP.mlp_torch_model import MultiLayerPerceptron
from lightning_model import LightningModel

from train_eval_network import train, eval
import lightning as L

from utils import LOG_DIR


def train_eval_mlp(train_dataloader, val_dataloader, test_dataloader, num_epochs, device):
    model = MultiLayerPerceptron(num_features=784, num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    train(model, optimizer, num_epochs, train_dataloader, val_dataloader, device)
    eval(model, test_dataloader, device)


def train_eval_mlp_lightning(train_dataloader, val_dataloader, test_dataloader, num_epochs):
    model = MultiLayerPerceptron(num_features=784, num_classes=10)
    lightning_model = LightningModel(model=model, lr=0.05, num_epochs=num_epochs)
    callbacks = [ModelCheckpoint(save_top_k=1, monitor="val_acc", mode="max", save_last=True)]
    trainer = L.Trainer(max_epochs=num_epochs,
                        accelerator="auto",
                        devices="auto",
                        deterministic=True,
                        callbacks=callbacks,
                        logger=CSVLogger(save_dir=LOG_DIR, name="mlp_mnist"))
    trainer.fit(model=lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=lightning_model, dataloaders=test_dataloader)


def load_lightning_model_and_test(test_dataloader, device):
    model = MultiLayerPerceptron(num_features=784, num_classes=10)
    lightning_model = LightningModel.load_from_checkpoint(model=model,
                                                          checkpoint_path="../Logs/mlp_mnist/version_0/checkpoints/last.ckpt",
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
    train_dataloader, val_dataloader, test_dataloader = init_dataloaders(flat=True)
    num_epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # train_eval_mlp(train_dataloader, val_dataloader, test_dataloader, num_epochs, device)
    # train_eval_mlp_lightning(train_dataloader, val_dataloader, test_dataloader, num_epochs)
    load_lightning_model_and_test(test_dataloader, device)
