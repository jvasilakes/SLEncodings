import os
import json
import warnings

import torch
import numpy as np
import pandas as pd
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
            X.append(np.array(example['x']))
            gold_y.append(onehot(example["preferred_y"], labeldim))
            for ann in example["annotations"]:
                Y[i].append(onehot(ann['y'], labeldim))
                metadata[i].append(
                        {"example_id": example["example_id"],
                         "annotator_id": ann["annotator_id"],
                         "annotator_reliability": ann["annotator_reliability"],
                         "annotator_certainty": ann["annotator_certainty"]}
                        )

        return {'X': X, 'Y': Y, "gold_y": gold_y, "metadata": metadata}


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
        train_i, other_i = train_test_split(idxs, train_size=0.7,
                                            shuffle=True, stratify=gold_y)
        val_i, test_i = train_test_split(other_i, test_size=(2/3),
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


@register("cifar10h")
class CIFAR10HDataLoader(object):
    """
    datadir: /path/to/dir/ containing cifar-10-python.tar.gz
    labelfile: /path/to/cifar10s_t2clamp_redist10.json
    """

    def __init__(self, datadir, labelfile):
        self.datadir = datadir
        self.labelfile = labelfile
        self.train, self.val, self.test = self.load()

    def load(self):
        transform_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)])
        transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform = transforms.Compose([transform_augment, transform_normalize])  # noqa
        # Per previous work we train on test and validate on train
        images_train = torchvision.datasets.CIFAR10(
            self.datadir, train=False, transform=transform)
        images_val = torchvision.datasets.CIFAR10(
            self.datadir, train=True, transform=transform)

        annotations = pd.read_csv(self.labelfile)
        X = []
        Y = []
        gold_y = []
        metadata = []
        for (i, anns) in annotations.groupby("cifar10_test_test_idx"):
            # Skip attention checks
            if i == -99999:
                continue
            X.append(images_train[i][0])
            Y.append([onehot(lab, 10) for lab in anns.chosen_label])
            gold_y.append(onehot(images_train[i][1], 10))
            metadata.append([{"example_id": i}
                             for _ in range(len(anns.chosen_label))])
        gold_y = np.array(gold_y)

        train_idxs = list(range(len(Y)))
        train_i, test_i = train_test_split(train_idxs, train_size=7000,
                                           shuffle=True, stratify=gold_y)

        val_labs = [ex[1] for ex in images_val]
        val_idxs = list(range(len(val_labs)))
        val_i, _ = train_test_split(val_idxs, train_size=3000,
                                    stratify=val_labs)
        valX = []
        valY = []
        val_metadata = []
        for i in val_i:
            img, lab = images_val[i]
            valX.append(img)
            valY.append([onehot(lab, 10)])
            val_metadata.append([{"example_id": i}])

        train = {'X': [X[i] for i in train_i],
                 'Y': [Y[i] for i in train_i],
                 "gold_y": gold_y[train_i],
                 "metadata": [metadata[i] for i in train_i]}
        val = {'X': valX,
               'Y': valY,
               "gold_y": valY,
               "metadata": val_metadata}
        test = {'X': [X[i] for i in test_i],
                'Y': [Y[i] for i in test_i],
                "gold_y": gold_y[test_i],
                "metadata": [metadata[i] for i in test_i]}
        return (train, val, test)


@register("cifar10")
class CIFAR10DataLoader(object):
    """
    datadir: /path/to/dir/ containing cifar-10-python.tar.gz
    labelfile: /path/to/cifar10s_t2clamp_redist10.json
    """

    def __init__(self, datadir):
        self.datadir = datadir
        self.train, self.val, self.test = self.load()

    def load(self):
        transform_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)])
        transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform = transforms.Compose([transform_augment, transform_normalize])  # noqa
        # Per previous work we train on test and validate on train
        images_train = torchvision.datasets.CIFAR10(
            self.datadir, train=False, transform=transform)
        images_val = torchvision.datasets.CIFAR10(
            self.datadir, train=True, transform=transform_normalize)
        images_test = torchvision.datasets.CIFAR10(
            self.datadir, train=False, transform=transform_normalize)

        train_labs = [ex[1] for ex in images_train]
        train_idxs = list(range(len(train_labs)))
        train_i, test_i = train_test_split(train_idxs, train_size=7000,
                                           shuffle=True, stratify=train_labs)

        val_labs = [ex[1] for ex in images_val]
        val_idxs = list(range(len(val_labs)))
        val_i, _ = train_test_split(val_idxs, train_size=3000,
                                    stratify=val_labs)

        train = self.get_data(train_i, images_train)
        val = self.get_data(val_i, images_val)
        test = self.get_data(test_i, images_test)
        return (train, val, test)

    def get_data(self, indices, images):
        X = []
        Y = []
        gold_y = []
        metadata = []
        for i in indices:
            img, lab = images[i]
            X.append(img)
            Y.append([onehot(lab, 10)])
            gold_y.append([onehot(lab, 10)])
            metadata.append([{"example_id": i}])
        return {'X': X, 'Y': Y, "gold_y": gold_y, "metadata": metadata}


if __name__ == "__main__":
    print("Available dataloaders:")
    for key in DATALOADER_REGISTRY.keys():
        print(f" * {key}")
