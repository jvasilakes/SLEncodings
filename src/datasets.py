import warnings

import torch
import numpy as np
from torch.utils.data import Dataset

import sle


class Annotator(object):

    instances = []

    def __init__(self, label_set, reliability=1.0, certainty=1.0):
        self.id = len(Annotator.instances)
        self.label_set = set(label_set)
        self.reliability = reliability
        self.certainty = certainty
        Annotator.instances.append(self)

    def annotate(self, true_label):
        agree = np.random.choice(
                [True, False],
                p=[self.reliability, 1. - self.reliability])
        if agree.item() is True:
            return true_label
        else:
            other_labels = list(self.label_set - set([true_label]))
            return np.random.choice(other_labels)


class MultiAnnotatorDataset(Dataset):
    """
    annotators: int or list of Annotator instances
    """
    def __init__(self, X, Y, annotator_params, gold_y=None):
        self.X = X
        self.Y = Y
        self.annotator_params = annotator_params
        self._gold_y = gold_y
        self.preprocess()
        self.aggregate_labels()

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}()"

    def __str__(self):
        name = self.__class__.__name__
        return f"{name}()"

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

    @property
    def labels(self):
        return set(self.Y.flatten())

    @property
    def gold_y(self):
        if self._gold_y is None:
            warnings.warn("No gold labels available.")
            return None
        else:
            return self._gold_y

    def preprocess(self):
        """
        Override in subclasses
        """
        for i in range(len(self.Y)):
            self.X[i] = torch.as_tensor(self.X[i], dtype=torch.float32)
            self.Y[i] = torch.as_tensor(self.Y[i], dtype=torch.float32)

    def aggregate_labels(self):
        """
        Override in subclasses
        """
        pass


class NonAggregatedDataset(MultiAnnotatorDataset):
    """
    Each annotation is a separate example.
    """

    def preprocess(self):
        new_X = []
        new_Y = []
        for (x, ys) in zip(self.X, self.Y):
            tileshape = torch.ones(len(x.shape)+1, dtype=int)
            tileshape[0] = len(ys)
            xs = torch.as_tensor(np.tile(x, tileshape), dtype=torch.float32)
            new_X.extend(xs)
            new_Y.extend(ys)
        self.X = new_X
        self.Y = new_Y


class VotingAggregatedDataset(MultiAnnotatorDataset):
    """
    Aggregate annotations according to majority voting.
    """

    def aggregate_labels(self):
        new_Y = []
        for ys in self.Y:
            y_idx = ys.argmax(axis=1).bincount().argmax()
            y_onehot = torch.zeros_like(ys[0])
            y_onehot[y_idx] = 1.
            new_Y.append(y_onehot)
        self.Y = new_Y


class SubjectiveLogicDataset(MultiAnnotatorDataset):
    """
    Encode individual annotations as Subjective Logic Encodings (SLEs).
    """

    def old_preprocess(self):
        label_dim = len(set([y_i for ys in self.Y for y_i in ys]))
        new_Y = []
        for (ys, ann_params) in zip(self.Y, self.annotator_params):
            uncertainties = None
            if "certainty" in ann_params[0].keys():
                uncertainties = [1.0 - ann["certainty"] for ann in ann_params]
            ys_enc = sle.encode_labels(
                    ys, label_dim, uncertainties=uncertainties)
            new_Y.append(ys_enc)
        self.Y = new_Y

    def preprocess(self):
        new_Y = []
        for (ys, ann_params) in zip(self.Y, self.annotator_params):
            uncertainties = None
            if "certainty" in ann_params[0].keys():
                uncertainties = [1.0 - ann["certainty"] for ann in ann_params]
            ys_enc = sle.encode_labels(ys, uncertainties=uncertainties)
            new_Y.append(ys_enc)
        self.Y = new_Y


class NonAggregatedSLDataset(SubjectiveLogicDataset):

    def preprocess(self):
        new_X = []
        new_Y = []
        for (x, ys, ann_params) in zip(self.X, self.Y, self.annotator_params):
            # Encode labels as SLEs
            uncertainties = None
            if "certainty" in ann_params[0].keys():
                uncertainties = [1.0 - ann["certainty"] for ann in ann_params]
            ys_enc = sle.encode_labels(ys, uncertainties=uncertainties)
            new_Y.extend(ys_enc)

            # Duplicate the inputs
            tileshape = torch.ones(len(x.shape)+1, dtype=int)
            tileshape[0] = len(ys)
            xs = torch.as_tensor(np.tile(x, tileshape), dtype=torch.float32)
            new_X.extend(xs)
        self.X = new_X
        self.Y = new_Y


class CumulativeFusionDataset(SubjectiveLogicDataset):
    """
    Aggregate SLEs using uncertainty-maximized cumulative fusion.
    """

    def aggregate_labels(self):
        new_Y = []
        for sle_ys in self.Y:
            fused = sle.fuse(sle_ys, max_uncertainty=True)
            new_Y.append(fused)
        self.Y = new_Y
