from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms


class MNIST(Dataset):
    def __init__(self, sequence: str = "train"):
        if sequence == "test":
            self.dataset = datasets.MNIST(root="./mnist", train=False, transform=transforms.ToTensor())
        else:
            self.dataset = datasets.MNIST(root="./mnist", train=True, transform=transforms.ToTensor(), download=True)
            if sequence == "val":
                _, self.dataset = random_split(self.dataset, lengths=[55000, 5000])

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


