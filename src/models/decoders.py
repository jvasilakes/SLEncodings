import torch.nn as nn


DECODER_REGISTRY = {}


def register_decoder(label_type):
    def add_to_registry(cls):
        DECODER_REGISTRY[label_type] = cls
        return cls
    return add_to_registry


@register_decoder("discrete")
class LinearDecoder(nn.Module):

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.threshold = 0.5

        if self.output_size == 1:
            self.activation = nn.Sigmoid()
            self._loss_fn = nn.BCELoss()
        else:
            self.activation = nn.Softmax(dim=1)
            self._loss_fn = nn.CrossEntropyLoss()

        self.predictor = nn.Sequential(
                nn.Linear(self.hidden_size, self.output_size),
                self.activation)

        self._opt = None

    def forward(self, encoded_inputs):
        return self.predictor(encoded_inputs)

    def predict(self, probs):
        """
        probs = self.forward(inputs)
        """
        preds = (probs >= self.threshold).int()
        return preds.argmax(axis=1)
