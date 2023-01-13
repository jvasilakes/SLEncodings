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
        self._lr_scheduler = None
        assert self.encoder.hidden_size == self.decoder.hidden_size

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

    def configure_optimizer(self):
        # Check for a task-specific optimizer
        if hasattr(self.encoder, "configure_optimizer"):
            opt_cls, opt_kwargs = self.encoder.configure_optimizer()
        else:
            # A good default option
            opt_cls = torch.optim.Adam
            opt_kwargs = {}
        return opt_cls, opt_kwargs

    def configure_lr_scheduler(self, *args):
        # Check for a learning rate scheduler
        if hasattr(self.encoder, "configure_lr_scheduler"):
            sched_cls, sched_kwargs, update_freq = self.encoder.configure_lr_scheduler(*args)  # noqa
        else:
            sched_cls, sched_kwargs, update_freq = (None, None, None)
        return sched_cls, sched_kwargs, update_freq
