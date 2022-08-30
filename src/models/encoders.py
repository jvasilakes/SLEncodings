import torch.nn as nn


ENCODER_REGISTRY = {}

def register_encoder(dataset_name):
    def add_to_registry(cls):
        ENCODER_REGISTRY[dataset_name] = cls
        return cls
    return add_to_registry


@register_encoder("synthetic")
class LinearEncoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
                nn.Linear(input_size, self.hidden_size),
                nn.Tanh())

    def forward(self, batch):
        inputs = batch[0]  # batch = (X, Y)
        return self.encoder(inputs)
