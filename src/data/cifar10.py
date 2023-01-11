import json

import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from src.data.utils import register, onehot


@register("cifar10")
class CIFAR10DataLoader(object):
    """
    datadir: /path/to/dir/ containing cifar-10-python.tar.gz
    labelfile: /path/to/cifar10s_t2clamp_redist10.json
    """

    def __init__(self, datadir, n_train=-1, random_seed=0, load=True,
                 train_idxs=None, val_idxs=None, test_idxs=None,
                 augment=True):
        self.datadir = datadir
        self.n_train = n_train
        self.random_seed = random_seed
        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs
        self.augment = augment
        self.base_train, self.base_test, self.base_val = self.load_cifar10base()  # noqa
        if load is True:
            self.train, self.val, self.test = self.load()

    def load_cifar10base(self):
        """
        Per previous work we train on test and validate on train.
        We only augment the training data.
        """
        transform_augment = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])
        transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        if self.augment is True:
            transform_train = transforms.Compose(
                [transform_augment, transform_normalize])
        else:
            transform_train = transform_normalize
        base_train = torchvision.datasets.CIFAR10(
            self.datadir, train=False, transform=transform_train)
        base_test = torchvision.datasets.CIFAR10(
            self.datadir, train=False, transform=transform_normalize)
        base_val = torchvision.datasets.CIFAR10(
            self.datadir, train=True, transform=transform_normalize)
        return (base_train, base_test, base_val)

    def load(self):
        labs = [ex[1] for ex in self.base_train]

        if self.train_idxs is None:
            idxs = list(range(len(labs)))
            self.train_idxs, self.test_idxs = train_test_split(
                idxs, train_size=7000, shuffle=True, stratify=labs,
                random_state=self.random_seed)

        if self.val_idxs is None:
            val_labs = [ex[1] for ex in self.base_val]
            val_idxs = list(range(len(val_labs)))
            self.val_idxs, _ = train_test_split(
                val_idxs, train_size=3000, shuffle=False, stratify=val_labs,
                random_state=self.random_seed)

        # Optionally load a subset of the training data.
        if self.n_train > 0:
            np.random.shuffle(self.train_idxs)
            self.train_idxs = self.train_idxs[:self.n_train]
        train = self.get_data(self.train_idxs, self.base_train)
        val = self.get_data(self.val_idxs, self.base_val)
        test = self.get_data(self.test_idxs, self.base_test)
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
            gold_y.append(onehot(lab, 10))
            metadata.append([{"example_id": i}])
        return {'X': X, 'Y': Y, "gold_y": gold_y, "metadata": metadata}


@register("cifar10s")
class CIFAR10SDataLoader(CIFAR10DataLoader):
    """
    datadir: /path/to/dir/ containing cifar-10-python.tar.gz
    labelfile: /path/to/cifar10s_t2clamp_redist10.json
    """

    def __init__(self, datadir, labelfile, n_train=-1, random_seed=0,
                 train_idxs=None, val_idxs=None, test_idxs=None):
        # No data augmentation works better on CIFAR10s.
        super().__init__(datadir, n_train=n_train,
                         random_seed=random_seed, load=False,
                         train_idxs=train_idxs, val_idxs=val_idxs,
                         test_idxs=test_idxs, augment=False)
        self.labelfile = labelfile
        self.train, self.val, self.test = self.load()

    def get_data(self, indices, images):
        labels_by_img = json.load(open(self.labelfile))
        # Convert string indices to int
        labels_by_img = {int(k): v for (k, v) in labels_by_img.items()}

        X = []
        Y = []
        gold_y = []
        metadata = []
        for i in indices:
            img, gold_lab = images[i]
            X.append(img)
            gold_lab = onehot(gold_lab, 10)
            gold_y.append(gold_lab)
            if i in labels_by_img.keys():
                # If this example is in CIFAR10S, use the soft labels
                no_na_labs = [lab for lab in labels_by_img[i]
                              if not np.isnan(lab).any()]
                labs = np.array(no_na_labs)
            else:
                # Otherwise use the gold label
                labs = np.array([gold_lab])
            Y.append(labs)
            metadata.append([{"example_id": i} for _ in range(len(labs))])
        return {'X': X, 'Y': Y, "gold_y": gold_y, "metadata": metadata}


@register("cifar10h")
class CIFAR10HDataLoader(CIFAR10DataLoader):
    """
    datadir: /path/to/dir/ containing cifar-10-python.tar.gz
    labelfile: /path/to/cifar10s_t2clamp_redist10.json
    """

    def __init__(self, datadir, labelfile, n_train=-1, random_seed=0,
                 train_idxs=None, val_idxs=None, test_idxs=None):
        super().__init__(datadir, n_train=n_train,
                         random_seed=random_seed, load=False,
                         train_idxs=train_idxs, val_idxs=val_idxs,
                         test_idxs=test_idxs, augment=True)
        self.labelfile = labelfile
        annotations = pd.read_csv(self.labelfile)
        self.labels_by_index = dict(
            (i, list(anns.chosen_label)) for (i, anns)
            in annotations.groupby("cifar10_test_test_idx")
            if i != -99999)  # Skip attention checks
        self.train, self.val, self.test = self.load()

    def get_data(self, indices, images):
        X = []
        Y = []
        gold_y = []
        metadata = []
        for i in indices:
            img, gold_lab = images[i]
            X.append(img)
            gold_lab = onehot(gold_lab, 10)
            gold_y.append(gold_lab)
            if i in self.labels_by_index.keys():
                labs = [onehot(lab, 10) for lab in self.labels_by_index[i]]
            else:
                labs = np.array([gold_lab])
            Y.append(labs)
            metadata.append([{"example_id": i} for _ in range(len(labs))])
        return {'X': X, 'Y': Y, "gold_y": gold_y, "metadata": metadata}
