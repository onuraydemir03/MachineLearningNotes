from collections import Counter
import os.path as op
from enum import Enum

import torch
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from CNN.cnn_torch_model import CNN
from CNN.resnet18 import Resnet18
from MLP.mlp_torch_model import MLP
from lightning_model import LightningModel
from train_eval_network import train, eval
import lightning as L

DATASETS_DIR = "/home/onuraydemir/Desktop/Code/DeepLearningFundamentalsClean/Datasets"
LOG_DIR = "/home/onuraydemir/Desktop/Code/DeepLearningFundamentalsClean/Logs"


class DefinedModels(Enum):
    MLP = 0
    CNN = 1
    RESNET18 = 2


def count_classes(dataloader: DataLoader):
    counter = Counter()
    for _, lbl in dataloader:
        counter.update(lbl.tolist())
    return counter


def get_majority_rule_acc(counter: Counter):
    majority_class = counter.most_common(1)[0]
    majority_rule_acc = majority_class[1] / sum(counter.values())
    return {majority_class: majority_rule_acc}


def train_eval(model, train_dataloader, val_dataloader, test_dataloader, num_epochs, device):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    train(model, optimizer, num_epochs, train_dataloader, val_dataloader, device)
    eval(model, test_dataloader, device)


def train_eval_lightning(model, train_dataloader, val_dataloader, test_dataloader, num_epochs, name):
    lightning_model = LightningModel(model=model, lr=0.05, num_epochs=num_epochs)
    callbacks = [ModelCheckpoint(save_top_k=1, monitor="val_acc", mode="max", save_last=True)]
    trainer = L.Trainer(max_epochs=num_epochs,
                        accelerator="auto",
                        devices="auto",
                        deterministic=True,
                        callbacks=callbacks,
                        logger=CSVLogger(save_dir=LOG_DIR, name=name))
    trainer.fit(model=lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=lightning_model, dataloaders=test_dataloader)


def load_lightning_model_and_test(model, test_dataloader, device, name, version_no):
    lightning_model = LightningModel.load_from_checkpoint(model=model,
                                                          checkpoint_path=op.join(LOG_DIR, name,
                                                                                  f"version_{version_no}",
                                                                                  "checkpoints", "last.ckpt"),
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


def get_model(model: DefinedModels,
              number_of_inputs: int,
              number_of_outputs: int):
    if model == DefinedModels.MLP:
        model = MLP(num_features=number_of_inputs, num_classes=number_of_outputs)
    elif model == DefinedModels.CNN:
        model = CNN(in_channels=number_of_inputs, num_classes=number_of_outputs)
    elif model == DefinedModels.RESNET18:
        model = Resnet18(num_classes=number_of_outputs)
    return model