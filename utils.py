from collections import Counter
from typing import Dict

from torch.utils.data import DataLoader


DATASETS_DIR = "/home/onuraydemir/Desktop/Code/DeepLearningFundamentalsClean/Datasets"
def count_classes(dataloader: DataLoader):
    counter = Counter()
    for _, lbl in dataloader:
        counter.update(lbl.tolist())
    return counter


def get_majority_rule_acc(counter: Counter):
    majority_class = counter.most_common(1)[0]
    majority_rule_acc = majority_class[1] / sum(counter.values())
    return {majority_class: majority_rule_acc}


