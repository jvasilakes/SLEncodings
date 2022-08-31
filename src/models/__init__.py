import torch
import torch.nn as nn

from .encoders import ENCODER_REGISTRY
from .decoders import DECODER_REGISTRY


def load_encoder_by_dataset_name(name, *args, **kwargs):
    return ENCODER_REGISTRY[name](*args, **kwargs)


def load_decoder_by_label_type(label_type, *args, **kwargs):
    return DECODER_REGISTRY[label_type](*args, **kwargs)


class CombinedModule(nn.Module):

    def __init__(self, encoder, decoder, lr=0.001):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self._opt = None

    def forward(self, batch):
        encoded = self.encoder(batch)
        return self.decoder(encoded)

    def compute_loss(self, outputs, batch):
        """
        outputs = self.foward(batch)

        batch contains the target. We use the whole batch
        here because different models/datasets may have different
        requirements for computing the loss and this keeps usage consistent.
        """
        return self.decoder.compute_loss(outputs, batch)

    def predict(self, outputs):
        """
        outputs = self.foward(batch)
        """
        return self.decoder.predict(outputs)

    @property
    def optimizer(self):
        if self._opt is None:
            self._opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return self._opt
