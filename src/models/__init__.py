import torch.nn as nn

from .encoders import ENCODER_REGISTRY
from .decoders import DECODER_REGISTRY


def load_encoder_by_dataset_name(name, *args, **kwargs):
    return ENCODER_REGISTRY[name](*args, **kwargs)


def load_decoder_by_label_type(label_type, *args, **kwargs):
    return DECODER_REGISTRY[label_type](*args, **kwargs)


class CombinedModule(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch):
        encoded = self.encoder(batch)
        return self.decoder(encoded)

    def compute_loss(self, outputs, target):
        """
        outputs = self.foward(batch)
        """
        return self.decoder._loss_fn(outputs, target)

    def predict(self, outputs):
        """
        outputs = self.foward(batch)
        """
        return self.decoder.predict(outputs)
