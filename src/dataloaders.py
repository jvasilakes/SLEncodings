import os
import json
import warnings

import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


DATALOADER_REGISTRY = {}


def register(name):
    def add_to_registry(cls):
        DATALOADER_REGISTRY[name] = cls
        return cls
    return add_to_registry


def load(dataset_name, *paths):
    dataset_name = dataset_name.lower()
    return DATALOADER_REGISTRY[dataset_name](*paths)


def onehot(y, ydim):
    onehot = np.zeros(ydim)
    onehot[y] = 1.
    return onehot.tolist()


@register("synthetic")
class SyntheticDataLoader(object):
    """
    dirpath: /path/to/directory containing {metadata,train,val,test}.json
    """

    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.metadata = self.load_metadata()
        self.train = self.load_split("train")
        self.val = self.load_split("val")
        self.test = self.load_split("test")

    def load_metadata(self):
        filepath = os.path.join(self.dirpath, "metadata.json")
        if os.path.isfile(filepath) is False:
            raise ValueError(f"No metadata.json found in {self.dirpath}!")
        return json.load(open(filepath))

    def load_split(self, split="train"):
        filepath = os.path.join(self.dirpath, f"{split}.json")
        if os.path.isfile(filepath) is False:
            warnings.warn("No {split}.json found!")
            return None
        examples = [json.loads(line) for line in open(filepath)]
        labeldim = self.metadata["n_classes"]

        X = []
        Y = [[] for _ in range(len(examples))]
        gold_y = []
        metadata = [[] for _ in range(len(examples))]

        for (i, example) in enumerate(examples):
            X.append(example['x'])
            gold_y.append(onehot(example["preferred_y"], labeldim))
            for ann in example["annotations"]:
                Y[i].append(onehot(ann['y'], labeldim))
                metadata[i].append(
                        {"example_id": example["example_id"],
                         "annotator_id": ann["annotator_id"],
                         "annotator_reliability": ann["annotator_reliability"],
                         "annotator_certainty": ann["annotator_certainty"]}
                        )

        return {'X': torch.as_tensor(X, dtype=torch.float32),
                'Y': torch.as_tensor(Y, dtype=torch.float32),
                "gold_y": torch.as_tensor(gold_y, dtype=torch.float32),
                "metadata": metadata}


@register("cifar10s")
class CIFAR10SDataLoader(object):
    """
    datadir: /path/to/dir/ containing cifar-10-python.tar.gz
    labelfile: /path/to/cifar10s_t2clamp_redist10.json
    """

    def __init__(self, datadir, labelfile):
        self.datadir = datadir
        self.labelfile = labelfile
        self.train, self.val, self.test = self.load()

    def load(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        images = torchvision.datasets.CIFAR10(self.datadir, train=False,
                                              transform=transform)

        labels_by_img = json.load(open(self.labelfile))
        X = []
        Y = []
        gold_y = []
        metadata = []
        for (str_idx, ys) in labels_by_img.items():
            i = int(str_idx)
            X.append(images[i][0])
            Y.append(np.array(ys))
            gold_y.append(onehot(images[i][1], 10))
            metadata.append([{"example_id": i} for _ in range(len(ys))])
        gold_y = np.array(gold_y)

        idxs = list(range(len(labels_by_img)))
        train_i, other_i = train_test_split(idxs, train_size=0.7, shuffle=True,
                                            random_state=0, stratify=gold_y)
        val_i, test_i = train_test_split(other_i, test_size=(2/3),
                                         random_state=0,
                                         stratify=gold_y[other_i])

        train = {'X': [X[i] for i in train_i],
                 'Y': [Y[i] for i in train_i],
                 "gold_y": gold_y[train_i],
                 "metadata": [metadata[i] for i in train_i]}
        val = {'X': [X[i] for i in val_i],
               'Y': [Y[i] for i in val_i],
               "gold_y": gold_y[val_i],
               "metadata": [metadata[i] for i in val_i]}
        test = {'X': [X[i] for i in test_i],
                'Y': [Y[i] for i in test_i],
                "gold_y": gold_y[test_i],
                "metadata": [metadata[i] for i in test_i]}
        return (train, val, test)


if __name__ == "__main__":
    print("Available dataloaders:")
    for key in DATALOADER_REGISTRY.keys():
        print(f" * {key}")
