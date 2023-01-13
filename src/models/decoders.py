import torch.nn as nn
import torch.distributions as D

import sle


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
            self.activation = nn.Softmax(dim=-1)
            self._loss_fn = nn.CrossEntropyLoss(reduction="mean")

        self.predictor = nn.Linear(self.hidden_size, self.output_size)

        self._opt = None

    def forward(self, encoded_inputs):
        return self.predictor(encoded_inputs)

    def predict(self, logits):
        """
        logits = self.forward(inputs)
        """
        probs = self.activation(logits)
        if self.output_size == 1:
            preds = (probs >= self.threshold).int()
        else:
            preds = probs.argmax(axis=1)
        return preds

    def compute_loss(self, logits, batch):
        target = batch['Y'].argmax(axis=1)
        return self._loss_fn(logits, target)


@register_decoder("sle")
class SLEDecoder(nn.Module):

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self._loss_fn = D.kl_divergence

        self.predictor = sle.SLELayer(hidden_size, output_size)
        self._opt = None

    def forward(self, encoded_inputs):
        dists = self.predictor(encoded_inputs)
        return dists.max_uncertainty()

    def predict(self, dists):
        """
        dists = self.forward(inputs)
        distributions predicted by the SLELayer
        """
        preds = dists.mean.argmax(axis=1)
        return preds

    def compute_loss(self, dists, batch):
        target = batch['Y']
        # Forward KL
        loss = self._loss_fn(dists, target).mean()
        return loss
