import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms

from utils import count_classes, get_majority_rule_acc, DATASETS_DIR
import os.path as op

class MNIST(Dataset):
    def __init__(self, sequence: str = "train"):
        if sequence == "test":
            self.dataset = datasets.MNIST(root=op.join(DATASETS_DIR, "mnist"), train=False, transform=transforms.ToTensor())
        else:
            self.dataset = datasets.MNIST(root=op.join(DATASETS_DIR, "mnist"), train=True, transform=transforms.ToTensor(), download=True)
            if sequence == "val":
                _, self.dataset = random_split(self.dataset, lengths=[55000, 5000])

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


def init_dataloaders():
    train_dataset = MNIST(sequence="train")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

    val_dataset = MNIST(sequence="val")
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    test_dataset = MNIST(sequence="test")
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


# if __name__ == '__main__':
#     torch.manual_seed(123)
#     train_dataset = MNIST(sequence="train")
#     test_dataset = MNIST(sequence="test")
#     train_dataloader = DataLoader(train_dataset, batch_size=64)
#     test_dataloader = DataLoader(test_dataset, batch_size=64)
#     train_counter = count_classes(train_dataloader)
#     test_counter = count_classes(test_dataloader)
#
#     majority_rule = get_majority_rule_acc(test_counter)

