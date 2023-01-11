import os
import json
import warnings

import numpy as np

from src.data.utils import register, onehot


@register("synthetic")
class SyntheticDataLoader(object):
    """
    dirpath: /path/to/directory containing {metadata,train,val,test}.json
    """

    def __init__(self, dirpath, n_train=-1, random_seed=0):
        self.dirpath = dirpath
        self.n_train = n_train
        self.random_seed = random_seed
        self.metadata = self.load_metadata()
        self.train = self.load_split("train", n=n_train)
        self.val = self.load_split("val")
        self.test = self.load_split("test")

    def load_metadata(self):
        filepath = os.path.join(self.dirpath, "metadata.json")
        if os.path.isfile(filepath) is False:
            raise ValueError(f"No metadata.json found in {self.dirpath}!")
        return json.load(open(filepath))

    def load_split(self, split="train", n=-1):
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

        if n > 0:
            return {'X': X[:n], 'Y': Y[:n],
                    "gold_y": gold_y[:n], "metadata": metadata[:n]}
        return {'X': X, 'Y': Y, "gold_y": gold_y, "metadata": metadata}
