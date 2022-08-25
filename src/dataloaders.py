import os
import json
import warnings

import numpy as np


DATALOADER_REGISTRY = {}


def register(name):
    def add_to_registry(cls):
        DATALOADER_REGISTRY[name] = cls
        return cls
    return add_to_registry


def load(dataset_name, path):
    dataset_name = dataset_name.lower()
    return DATALOADER_REGISTRY[dataset_name](path)


@register("synthetic")
class SyntheticDataLoader(object):

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

        X = []
        n_annotators = self.metadata["n_annotators"]
        Y = [np.zeros(n_annotators, dtype=int)
             for _ in range(len(examples))]
        gold_y = []
        annotator_params = [[] for _ in range(len(examples))]

        for (i, example) in enumerate(examples):
            X.append(np.array(example['x']))
            gold_y.append(example["preferred_y"])  # preferred is always first
            annotator_params[i].append(
                    {"id": "gold", "reliability": 1.0, "certainty": 1.0})
            for ann in example["annotations"]:
                ann_id = ann["annotator_id"]
                Y[i][ann_id] = ann['y']
                annotator_params[i].append(
                        {"id": ann["annotator_id"],
                         "reliability": ann["annotator_reliability"],
                         "certainty": ann["annotator_certainty"]}
                        )

        return {"X": np.array(X), 'Y': Y,
                "gold_y": np.array(gold_y),
                "annotator_params": annotator_params}


if __name__ == "__main__":
    print("Available dataloaders:")
    for key in DATALOADER_REGISTRY.keys():
        print(f" * {key}")
