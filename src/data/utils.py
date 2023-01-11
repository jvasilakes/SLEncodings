import numpy as np


DATALOADER_REGISTRY = {}


def register(name):
    def add_to_registry(cls):
        DATALOADER_REGISTRY[name] = cls
        return cls
    return add_to_registry


def load(dataset_name, *paths, **kwargs):
    dataset_name = dataset_name.lower()
    return DATALOADER_REGISTRY[dataset_name](*paths, **kwargs)


def onehot(y, ydim):
    assert isinstance(y, (int, np.integer)), f"y: {y}, {type(y)}"
    assert y >= 0
    vec = np.zeros(ydim)
    vec[y] = 1.
    return vec.tolist()
