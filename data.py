import json
from hashlib import md5

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from torch.utils.data import Dataset

from sle import SLBeta


class Annotator(object):

    instances = []

    def __init__(self, label_set, trustworthiness=1.0):
        self.label_set = set(label_set)
        self.trustworthiness = trustworthiness
        self.id = len(Annotator.instances)
        Annotator.instances.append(self)

    def annotate(self, true_label):
        agree = np.random.choice(
                [True, False],
                p=[self.trustworthiness, 1. - self.trustworthiness])
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

    def __init__(self, n_features, n_examples, annotators=10,
                 trustworthiness="perfect", random_seed=0, data=None):
        self.n_features = n_features
        self.n_examples = n_examples

        self.trustworthiness = trustworthiness
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
        return f"""{name}(n_features={self.n_features}, n_examples={self.n_examples}, annotators={len(self.annotators)}, trustworthiness={self.trustworthiness})"""  # noqa

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def subset(self, idxs):
        cls = self.__class__
        return cls(self.n_features, len(idxs),
                   annotators=self.annotators,
                   trustworthiness=self.trustworthiness,
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
        if self.trustworthiness == "perfect":
            trustworthinesses = np.ones(n)
        elif self.trustworthiness == "high":
            trustworthinesses = np.random.beta(10, 1, size=n)
        elif self.trustworthiness == "medium":
            trustworthinesses = np.random.beta(10, 10, size=n)
        elif self.trustworthiness == "low":
            trustworthinesses = np.random.beta(1, 10, size=n)
        elif self.trustworthiness == "high-outlier":
            trustworthinesses = np.random.beta(10, 1, size=n)
            trustworthinesses[0] = 0.1
        elif self.trustworthiness == "low-outlier":
            trustworthinesses = np.random.beta(1, 10, size=n)
            trustworthinesses[0] = 0.9
        else:
            raise ValueError(f"Unsupported trustworthiness '{self.trustworthiness}'")  # noqa
        return [Annotator(self.labels, trustworthinesses[i])
                for i in range(n)]

    def generate_data(self):
        np.random.seed(self.random_seed)
        X, y = make_classification(self.n_examples, self.n_features,
                                   n_classes=2, n_clusters_per_class=2,
                                   class_sep=1., flip_y=0.02,
                                   random_state=self.random_seed)
        data = []
        for (i, (x_i, pref_y)) in enumerate(zip(X, y)):
            for (j, annotator) in enumerate(self.annotators):
                ann_y = annotator.annotate(pref_y)
                uuid = md5(f"{i}{j}".encode()).hexdigest()
                data.append({"uuid": uuid, "example_id": i,
                             "annotator_id": annotator.id,
                             "annotator_trustworthiness": annotator.trustworthiness,  # noqa
                             "preferred_y": pref_y,
                             "x": x_i, "y": ann_y})

        data = self.add_uncertainties(data)
        return data

    def add_uncertainties(self, data):
        X = np.array([e['x'] for e in data if e["annotator_id"] == 0])
        y = np.array([e['preferred_y'] for e in data if e["annotator_id"] == 0])  # noqa
        svc = LinearSVC().fit(X, y)
        scores = np.abs(svc.decision_function(X))
        scores = scores / np.max(scores)  # normalize to [0,1]

        score_mean = np.mean(scores)
        score_sd = np.std(scores)
        unc_labels_thresholds = [(0.0, "uncertain"),
                                 (score_mean - score_sd, "somewhat_certain"),
                                 (score_mean, "certain")]
        for example in data:
            score = scores[example["example_id"]]
            label = None
            for (t, lab) in unc_labels_thresholds:
                if score >= t:
                    label = lab
            switch_unc_label = np.random.choice([True, False], p=[0.9, 0.1])
            if switch_unc_label is True:
                other_labs = [lab for lab in unc_labels_thresholds
                              if lab != label]
                label = np.random.choice(other_labs)
            example["certainty_level"] = label
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
                    "trustworthiness": self.trustworthiness,  # noqa
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
        colors = [color_map[ex["certainty_level"]] for ex in self]
        plt.scatter(X_emb[:, 0], X_emb[:, 1], c=colors, alpha=0.3)
        plt.xticks([])
        plt.yticks([])
        if savepath is not None:
            plt.savefig(savepath, dpi=300)
        else:
            plt.show()


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
            datum_cp["annotator_trustworthiness"] = [ex["annotator_trustworthiness"]  # noqa
                                                     for ex in example_group]
            datum_cp["certainty_level"] = [ex["certainty_level"]
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
            datum_cp["annotator_trustworthiness"] = [ex["annotator_trustworthiness"]  # noqa
                                                     for ex in example_group]
            datum_cp["certainty_level"] = [ex["certainty_level"]
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
            datum_cp["annotator_trustworthiness"] = [ex["annotator_trustworthiness"]  # noqa
                                                     for ex in example_group]
            datum_cp["certainty_level"] = [ex["certainty_level"]
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
                    "trustworthiness": self.trustworthiness,
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

    def plot(self, savepath=None):
        X = [ex['x'].numpy() for ex in self]
        Y = [ex['y'].mean for ex in self]
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
        colors = [color_map[ex["certainty_level"]] for ex in self]
        plt.scatter(X_emb[:, 0], X_emb[:, 1], c=colors, alpha=0.3)
        plt.xticks([])
        plt.yticks([])
        if savepath is not None:
            plt.savefig(savepath, dpi=300)
        else:
            plt.show()


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
            datum_cp["annotator_trustworthiness"] = [ex["annotator_trustworthiness"]  # noqa
                                                     for ex in example_group]
            datum_cp["certainty_level"] = [ex["certainty_level"]
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
            datum_cp["annotator_trustworthiness"] = [ex["annotator_trustworthiness"]  # noqa
                                                     for ex in example_group]
            datum_cp["certainty_level"] = [ex["certainty_level"]
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
        colors = [color_map[ex["certainty_level"]] for ex in self]
        plt.scatter(X_emb[:, 0], X_emb[:, 1], c=colors, alpha=0.3)
        plt.xticks([])
        plt.yticks([])
        if savepath is not None:
            plt.savefig(savepath, dpi=300)
        else:
            plt.show()
