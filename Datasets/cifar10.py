import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from utils import count_classes, get_majority_rule_acc, DATASETS_DIR
import os.path as op
import torchvision.transforms as T


class Cifar10(Dataset):
    def __init__(self, sequence: str = "train", train_transforms=None, test_transforms=None):
        if train_transforms is None:
            train_transforms = T.Compose([
                T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
                T.RandomCrop((224, 224)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        if test_transforms is None:
            test_transforms = T.Compose([
                T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop((224, 224)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        if sequence == "test":
            self.dataset = datasets.CIFAR10(root=op.join(DATASETS_DIR, "cifar10"),
                                            train=False,
                                            transform=test_transforms)
        else:
            self.dataset = datasets.CIFAR10(root=op.join(DATASETS_DIR, "cifar10"),
                                            train=True,
                                            transform=train_transforms, download=True)
            if sequence == "val":
                _, self.dataset = random_split(self.dataset, lengths=[45000, 5000])

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


def init_dataloaders():
    train_dataset = Cifar10(sequence="train")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

    val_dataset = Cifar10(sequence="val")
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    test_dataset = Cifar10(sequence="test")
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def init_dataloaders(batch_size: int = 64):
    train_dataset = Cifar10(sequence="train")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = Cifar10(sequence="val")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = Cifar10(sequence="test")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    torch.manual_seed(123)
    train_dataset = Cifar10(sequence="train", train_transforms=T.ToTensor(), test_transforms=T.ToTensor())
    test_dataset = Cifar10(sequence="test", train_transforms=T.ToTensor(), test_transforms=T.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=512)
    test_dataloader = DataLoader(test_dataset, batch_size=512)
    train_counter = count_classes(train_dataloader)
    test_counter = count_classes(test_dataloader)

    majority_rule = get_majority_rule_acc(test_counter)

