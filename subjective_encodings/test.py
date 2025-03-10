import torch
import torch.distributions as D

import sle
from sle.collate import collate_sle_labels


def test_binary_one_example():
    N = 1
    input_dim = 10
    hidden_dim = 5
    label_dim = 2
    x = torch.randn((N, input_dim))
    y = torch.randint(label_dim, size=(N, 1))
    encoded = sle.encode_labels(y)
    encoded = collate_sle_labels(encoded)
    model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            sle.SLELayer(hidden_dim, label_dim))
    outputs = model(x)
    loss = D.kl_divergence(outputs, encoded)
    print("y:\t ", y.shape)
    print("encoded:\t ", encoded.b.shape)
    print("outputs:\t", outputs.b.shape)
    print("KL:\t", loss.shape)


def test_binary_N_examples():
    N = 4
    input_dim = 10
    hidden_dim = 5
    label_dim = 2
    x = torch.randn((N, input_dim))
    y = torch.randint(label_dim, size=(N, 1))
    encoded = sle.encode_labels(y)
    encoded = collate_sle_labels(encoded)
    model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            sle.SLELayer(hidden_dim, label_dim))
    outputs = model(x)
    loss = D.kl_divergence(outputs, encoded)
    print("y:\t ", y.shape)
    print("encoded:\t ", encoded.b.shape)
    print("outputs:\t", outputs.b.shape)
    print("KL:\t", loss.shape)


def test_multiclass_one_example():
    N = 1
    input_dim = 10
    hidden_dim = 5
    label_dim = 3
    x = torch.randn((N, input_dim))
    y_idxs = torch.randint(label_dim, size=(N,))
    y = torch.zeros((N, label_dim))
    y[torch.arange(N), y_idxs] = 1.  # one-hot encoding
    encoded = sle.encode_labels(y)
    encoded = collate_sle_labels(encoded)
    model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            sle.SLELayer(hidden_dim, label_dim))
    outputs = model(x)
    loss = D.kl_divergence(outputs, encoded)
    print("y:\t ", y.shape)
    print("encoded:\t ", encoded.b.shape)
    print("outputs:\t", outputs.b.shape)
    print("KL:\t", loss.shape)


def test_multiclass_N_examples():
    N = 4
    input_dim = 10
    hidden_dim = 5
    label_dim = 3
    x = torch.randn((N, input_dim))
    y_idxs = torch.randint(label_dim, size=(N,))
    y = torch.zeros((N, label_dim))
    y[torch.arange(N), y_idxs] = 1.  # one-hot encoding
    encoded = sle.encode_labels(y)
    encoded = collate_sle_labels(encoded)
    model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            sle.SLELayer(hidden_dim, label_dim))
    outputs = model(x)
    loss = D.kl_divergence(outputs, encoded)
    print("y:\t ", y.shape)
    print("encoded:\t ", encoded.b.shape)
    print("outputs:\t", outputs.b.shape)
    print("KL:\t", loss.shape)


if __name__ == "__main__":
    print("test_binary_one_example()")
    test_binary_one_example()
    print()

    print("test_binary_N_examples()")
    test_binary_N_examples()
    print()

    print("test_multiclass_one_example()")
    test_multiclass_one_example()
    print()

    print("test_multiclass_N_examples()")
    test_multiclass_N_examples()
    print()
