import numpy as np


def irs(example_group):
    """
    Item relation score
    """
    raise NotImplementedError()


def ct_precision_recall_fscore(ytrue, ypred, example_ids):
    """
    Assumes there are duplicate examples with different judgments
    in ytrue, which can be grouped by example_ids.
    """
    raise NotImplementedError()


def cross_entropy(ytrue, ypred):
    return -np.sum(ytrue * np.log(ypred), axis=1)


def js_divergence():
    """
    JSD(p_a(x) || p_b(x)) = 0.5 * KLD(p_a(x) || M) + 0.5 * KLD(p_b(x) || M)
    M = (p_a(x) + p_b(x)) / 2
    """
    raise NotImplementedError()


def entropy_similarity():
    """
    Cosine similarity between vectors of H_norm(p(x_i))
    and H_norm(p_hum(x_i)).
    """
    raise NotImplementedError()


def entropy_correlation():
    """
    Pearson's correlation between vectors of H_norm(p(x_i))
    and H_norm(p_hum(x_i)).
    """
    raise NotImplementedError()
