import torch
import torch.distributions as D

import subjective_encodings as sle


def test_binary_one_example():
    N = 1
    input_dim = 10
    hidden_dim = 5
    label_dim = 2
    x = torch.randn((N, input_dim))
    y = torch.randint(label_dim, size=(N,1))
    encoded = sle.encode_labels(y, label_dim, collate=True)
    model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            sle.BetaLayer(hidden_dim))
    outputs = model(x)
    loss = D.kl_divergence(outputs, encoded)
    print("y:\t ", y.shape)
    print("encoded:\t ", encoded.b.shape)
    print("outputs:\t", outputs.b.shape)
    print("loss:\t", loss.shape)


def test_binary_N_examples():
    N = 4
    input_dim = 10
    hidden_dim = 5
    label_dim = 2
    x = torch.randn((N, input_dim))
    y = torch.randint(label_dim, size=(N,1))
    encoded = sle.encode_labels(y, label_dim, collate=True)
    model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            sle.BetaLayer(hidden_dim))
    outputs = model(x)
    loss = D.kl_divergence(outputs, encoded)
    print("y:\t ", y.shape)
    print("encoded:\t ", encoded.b.shape)
    print("outputs:\t", outputs.b.shape)
    print("loss:\t", loss.shape)


def test_multiclass_one_example():
    N = 1
    input_dim = 10
    hidden_dim = 5
    label_dim = 3
    x = torch.randn((N, input_dim))
    y = torch.randint(label_dim, size=(N,1))
    encoded = sle.encode_labels(y, label_dim, collate=True)
    model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            sle.DirichletLayer(hidden_dim, label_dim))
    outputs = model(x)
    loss = D.kl_divergence(outputs, encoded)
    print("y:\t ", y.shape)
    print("encoded:\t ", encoded.b.shape)
    print("outputs:\t", outputs.b.shape)
    print("loss:\t", loss.shape)


def test_multiclass_N_examples():
    N = 4
    input_dim = 10
    hidden_dim = 5
    label_dim = 3
    x = torch.randn((N, input_dim))
    y = torch.randint(label_dim, size=(N,1))
    encoded = sle.encode_labels(y, label_dim, collate=True)
    model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            sle.DirichletLayer(hidden_dim, label_dim))
    outputs = model(x)
    loss = D.kl_divergence(outputs, encoded)
    print("y:\t ", y.shape)
    print("encoded:\t ", encoded.b.shape)
    print("outputs:\t", outputs.b.shape)
    print("loss:\t", loss.shape)


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
