import json
from hashlib import md5

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.datasets import make_classification
from torch.utils.data import Dataset

from sle import SLBeta


def split_dataset(dataset):
    example_ids = list(set([ex["example_id"] for ex in dataset]))
    train_ids, eval_ids = train_test_split(
            example_ids, train_size=0.8, random_state=0)
    val_ids, test_ids = train_test_split(
            eval_ids, test_size=0.5, random_state=0)

    train_idxs = []
    val_idxs = []
    test_idxs = []
    for (i, ex) in enumerate(dataset):
        if ex["example_id"] in train_ids:
            train_idxs.append(i)
        elif ex["example_id"] in val_ids:
            val_idxs.append(i)
        elif ex["example_id"] in test_ids:
            test_idxs.append(i)
    assert len(train_idxs) + len(val_idxs) + len(test_idxs) == len(dataset)
    train_ds = dataset.subset(train_idxs)
    val_ds = dataset.subset(val_idxs)
    test_ds = dataset.subset(test_idxs)
    return train_ds, val_ds, test_ds


class Annotator(object):

    instances = []

    def __init__(self, label_set, reliability=1.0, confidence=1.0):
        self.id = len(Annotator.instances)
        self.label_set = set(label_set)
        self.reliability = reliability
        self.confidence = confidence
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

    @classmethod
    def from_file(cls, datapath, n=-1):
        data = json.load(open(datapath))
        examples = data["examples"]
        if n > 0:
            examples = examples[:n]
        return cls(**data["metadata"], data=examples)

    def __init__(self, n_examples, n_features, annotators=10,
                 reliability="perfect", confidence="perfect",
                 random_seed=0, data=None):
        self.n_examples = n_examples
        self.n_features = n_features
        self.reliability = reliability
        self.confidence = confidence

        if isinstance(annotators, int):
            self.annotators = self.get_annotators(annotators)
        elif all([isinstance(ann, Annotator) for ann in annotators]):
            self.annotators = annotators
        else:
            raise ValueError(f"annotators must be int or list(Annotators). Got {type(annotators)}.")  # noqa
        self.random_seed = random_seed
        if data is not None:
            self._data = data
        else:
            self._data = self.generate_data()
        self._data = self.preprocess_data(self._data)
        self._data = self.aggregate_labels(self._data)

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}()"

    def __str__(self):
        name = self.__class__.__name__
        return f"""{name}(n_features={self.n_features}, n_examples={self.n_examples}, annotators={len(self.annotators)}, reliability={self.reliability})"""  # noqa

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def subset(self, idxs):
        cls = self.__class__
        return cls(self.n_features, len(idxs),
                   annotators=self.annotators,
                   reliability=self.reliability,
                   random_seed=self.random_seed,
                   data=[self[i] for i in idxs])

    @property
    def label_dim(self):
        # TODO: allow beyond binary
        return 1

    @property
    def labels(self):
        return [0., 1.]

    @property
    def examples(self):
        example_ids = sorted(set([ex["example_id"] for ex in self]))
        for eid in example_ids:
            group = [ex for ex in self if ex["example_id"] == eid]
            yield group

    def get_annotators(self, n):
        params_map = {"perfect": (10, 1e-18),
                      "high": (10, 1),
                      "high-outlier": (10, 1),
                      "medium": (10, 10),
                      "low": (1, 10),
                      "low-outlier": (1, 10)
                      }

        reliability_params = params_map[self.reliability]
        reliabilities = np.random.beta(*reliability_params, size=n)
        if self.reliability == "high-outlier":
            reliabilities[0] = 0.1
        elif self.reliability == "low-outlier":
            reliabilities[0] = 0.9

        confidence_params = params_map[self.confidence]
        confidences = np.random.beta(*confidence_params, size=n)
        if self.confidence == "high-outlier":
            confidences[0] = 0.1
        elif self.confidence == "low-outlier":
            confidences[0] = 0.9

        return [Annotator(self.labels, reliabilities[i], confidences[i])
                for i in range(n)]

    def generate_data(self):
        np.random.seed(self.random_seed)
        X, y = make_classification(self.n_examples, self.n_features,
                                   n_classes=3, n_clusters_per_class=1,
                                   n_redundant=0, class_sep=1.5, flip_y=0.01,
                                   random_state=self.random_seed)
        data = []
        for (i, (x_i, pref_y)) in enumerate(zip(X, y)):
            for (j, annotator) in enumerate(self.annotators):
                ann_y = annotator.annotate(pref_y)
                uuid = md5(f"{i}{j}".encode()).hexdigest()
                data.append({"uuid": uuid, "example_id": i,
                             "annotator_id": annotator.id,
                             "annotator_reliability": annotator.reliability,
                             "annotator_confidence": annotator.confidence,
                             "preferred_y": pref_y,
                             "x": x_i, "y": ann_y})
        return data

    def preprocess_data(self, data):
        for datum in data:
            datum['x'] = torch.as_tensor(datum['x'], dtype=torch.float32)
            datum['y'] = torch.as_tensor(datum['y'], dtype=torch.float32)
            datum["preferred_y"] = torch.as_tensor(
                    datum["preferred_y"], dtype=torch.float32)
        return data

    def aggregate_labels(self, data):
        return data

    def save(self, outpath):
        metadata = {"n_features": self.n_features,
                    "n_examples": self.n_examples,
                    "annotators": len(self.annotators),
                    "reliability": self.reliability,
                    "confidence": self.confidence,
                    "random_seed": self.random_seed}
        non_tensor_data = []
        for datum in self._data:
            datum_cp = dict(datum)
            datum_cp['x'] = datum_cp['x'].tolist()
            datum_cp['y'] = datum_cp['y'].item()
            datum_cp["preferred_y"] = datum_cp["preferred_y"].item()
            non_tensor_data.append(datum_cp)
        outdata = {"metadata": metadata,
                   "examples": non_tensor_data}
        with open(outpath, 'w') as outF:
            json.dump(outdata, outF)


class VotingAggregatedDataset(MultiAnnotatorDataset):

    def aggregate_labels(self, data):
        new_data = []
        for example_group in self.examples:
            ys = [ex['y'] for ex in example_group]
            y = np.bincount(ys).argmax()
            y = torch.tensor(y, dtype=torch.float32)
            datum_cp = dict(example_group[0])
            datum_cp['y'] = y
            datum_cp["annotator_id"] = [ex["annotator_id"]
                                        for ex in example_group]
            datum_cp["annotator_reliability"] = [ex["annotator_reliability"]  # noqa
                                                 for ex in example_group]
            datum_cp["confidence"] = [ex["confidence"]
                                      for ex in example_group]
            new_data.append(datum_cp)
        return new_data


class FrequencyAggregatedDataset(MultiAnnotatorDataset):

    def aggregate_labels(self, data):
        new_data = []
        for example_group in self.examples:
            ys = [ex['y'] for ex in example_group]
            y = np.bincount(ys) / len(ys)
            y = y[-1]
            datum_cp = dict(example_group[0])
            datum_cp['y'] = torch.tensor(y, dtype=torch.float32)
            datum_cp["annotator_id"] = [ex["annotator_id"]
                                        for ex in example_group]
            datum_cp["annotator_reliability"] = [ex["annotator_reliability"]  # noqa
                                                 for ex in example_group]
            datum_cp["confidence"] = [ex["confidence"]
                                      for ex in example_group]
            new_data.append(datum_cp)
        return new_data


class CatSampleAggregatedDataset(MultiAnnotatorDataset):

    def aggregate_labels(self, data):
        new_data = []
        for example_group in self.examples:
            ys = [ex['y'] for ex in example_group]
            ps = np.bincount(ys) / len(ys)
            if len(ps) == 1:
                ps = np.array([1., 0.])
            y = np.random.choice([0., 1.], p=ps)
            datum_cp = dict(example_group[0])
            datum_cp['y'] = torch.tensor(y, dtype=torch.float32)
            datum_cp["annotator_id"] = [ex["annotator_id"]
                                        for ex in example_group]
            datum_cp["annotator_reliability"] = [ex["annotator_reliability"]  # noqa
                                                 for ex in example_group]
            datum_cp["confidence"] = [ex["confidence"]
                                      for ex in example_group]
            new_data.append(datum_cp)
        return new_data


class SubjectiveLogicDataset(MultiAnnotatorDataset):

    @classmethod
    def from_file(cls, datapath, n=-1):
        data = json.load(open(datapath))
        examples = data["examples"]
        if n > 0:
            examples = examples[:n]
        return cls(**data["metadata"], data=examples)

    @staticmethod
    def label2beta(label):
        if label not in [0, 1]:
            raise ValueError("Only binary {0,1} values are supported")

        EPS = 1e-18
        u = torch.tensor(0. + EPS)
        if label == 1:
            b = torch.tensor(1. - EPS)
            d = torch.tensor(0.)
        else:
            b = torch.tensor(0.)
            d = torch.tensor(1. - EPS)
        return SLBeta(b, d, u)

    def beta2label(self, beta):
        return (beta.sample() >= 0.5).int().item()

    def preprocess_data(self, data):
        for datum in data:
            datum['x'] = torch.tensor(datum['x'], dtype=torch.float32)
            datum['y'] = self.label2beta(datum['y'])
            datum["preferred_y"] = torch.tensor(
                    datum["preferred_y"], dtype=torch.float32)
        return data

    def save(self, outpath):
        metadata = {"n_features": self.n_features,
                    "n_examples": self.n_examples,
                    "annotators": len(self.annotators),
                    "reliability": self.reliability,
                    "confidence": self.confidence,
                    "random_seed": self.random_seed}
        non_tensor_data = []
        for datum in self._data:
            datum_cp = dict(datum)
            datum_cp['x'] = datum_cp['x'].tolist()
            datum_cp['y'] = self.beta2label(datum_cp['y'])
            datum_cp["preferred_y"] = datum_cp["preferred_y"].item()
            non_tensor_data.append(datum_cp)
        outdata = {"metadata": metadata,
                   "examples": non_tensor_data}
        with open(outpath, 'w') as outF:
            json.dump(outdata, outF)


class CumulativeFusionDataset(SubjectiveLogicDataset):

    def aggregate_labels(self, data):
        new_data = []
        for example_group in self.examples:
            ys = [ex['y'] for ex in example_group]
            y = ys[0]
            for y_i in ys[1:]:
                y = y.cumulative_fusion(y_i)
            y = y.max_uncertainty()
            datum_cp = dict(example_group[0])
            datum_cp['y'] = y
            datum_cp["annotator_id"] = [ex["annotator_id"]
                                        for ex in example_group]
            datum_cp["annotator_reliability"] = [ex["annotator_reliability"]  # noqa
                                                 for ex in example_group]
            datum_cp["confidence"] = [ex["confidence"]
                                      for ex in example_group]
            new_data.append(datum_cp)
        return new_data


class SLSampleAggregatedDataset(SubjectiveLogicDataset):

    def aggregate_labels(self, data):
        new_data = []
        for example_group in self.examples:
            dists = [ex['y'] for ex in example_group]
            dist = dists[0]
            for d_i in dists[1:]:
                dist = dist.cumulative_fusion(d_i)
            dist = dist.max_uncertainty()
            y = dist.sample()
            datum_cp = dict(example_group[0])
            datum_cp['y'] = y
            datum_cp["annotator_id"] = [ex["annotator_id"]
                                        for ex in example_group]
            datum_cp["annotator_reliability"] = [ex["annotator_reliability"]  # noqa
                                                 for ex in example_group]
            datum_cp["confidence"] = [ex["confidence"]
                                      for ex in example_group]
            new_data.append(datum_cp)
        return new_data

    def plot(self, savepath=None):
        X = [ex['x'].numpy() for ex in self]
        Y = [ex['y'].item() for ex in self]
        X_emb = TSNE(n_components=2).fit_transform(X)

        plt.figure(figsize=(18, 9))
        plt.subplot(1, 2, 1)
        color_map = {1.: "#af8dc3", 0.: "#7fbf7b"}
        colors = [color_map[y] for y in Y]
        plt.scatter(X_emb[:, 0], X_emb[:, 1], c=colors, alpha=0.3)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 2, 2)
        color_map = {"certain": "#66c2a5",
                     "somewhat_certain": "#8da0cb",
                     "uncertain": "#fc8d62"}
        colors = [color_map[ex["confidence"]] for ex in self]
        plt.scatter(X_emb[:, 0], X_emb[:, 1], c=colors, alpha=0.3)
        plt.xticks([])
        plt.yticks([])
        if savepath is not None:
            plt.savefig(savepath, dpi=300)
        else:
            plt.show()
