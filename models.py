import torch
import torch.nn as nn
import torch.distributions as D

from distributions import SLBeta
from layers import BetaLayer, DirichletLayer


class LinearNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.threshold = 0.5
        self.lr = lr

        self.encoder = nn.Sequential(
                nn.Linear(input_size, self.hidden_size),
                nn.Tanh())
        self.predictor = nn.Sequential(
                nn.Linear(hidden_size, self.output_size),
                nn.Sigmoid())

        self._loss_fn = nn.BCELoss()
        self._opt = None

    def __repr__(self):
        return f"LinearNet(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size}, lr={self.lr})"  # noqa

    def __str__(self):
        return f"LinearNet(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size}, lr={self.lr})"  # noqa

    def forward(self, batch):
        encoded = self.encoder(batch)
        probs = self.predictor(encoded)
        return {"probs": probs}

    @property
    def optimizer(self):
        if self._opt is None:
            self._opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return self._opt

    def compute_loss(self, output, _input):
        y = _input['y']
        if y.dim() == 0:
            y = y.unsqueeze(0)
        return self._loss_fn(output["probs"], y)


class SLNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.threshold = 0.5
        self.lr = lr

        self.encoder = nn.Sequential(
                nn.Linear(input_size, self.hidden_size),
                nn.Tanh())
        # TODO: allow use of Dirichlet for multi-label
        self.params = nn.Sequential(
                nn.Linear(hidden_size, 3),
                nn.Softmax(dim=0))
        self._loss_fn = D.kl_divergence
        self._opt = None

    def __repr__(self):
        return f"SLNet(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size}, lr={self.lr})"  # noqa

    def __str__(self):
        return f"SLNet(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size}, lr={self.lr})"  # noqa

    def forward(self, batch):
        encoded = self.encoder(batch)
        belief, disbelief, unc = self.params(encoded)
        beta_dist = SLBeta(belief, disbelief, unc).max_uncertainty()
        return {"distribution": beta_dist}

    @property
    def optimizer(self):
        if self._opt is None:
            self._opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return self._opt

    def compute_loss(self, output, _input):
        return self._loss_fn(output["distribution"], _input['y'])


class AggregatingSLNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.threshold = 0.5
        self.lr = lr

        self.encoder = nn.Sequential(
                nn.Linear(input_size, self.hidden_size),
                nn.Tanh())
        if self.output_size in [1, 2]:
            self.output_layer = BetaLayer(self.hidden_size)
        elif self.output_size > 2:
            self.output_layer = DirichletLayer(self.hidden_size, self.outsize)
        else:
            raise ValueError(f"output_size must be > 0. Got {self.output_size}.")  # noqa

        self._loss_fn = D.kl_divergence
        self._opt = None

    def __repr__(self):
        return f"AggregatingSLNet(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size}, lr={self.lr})"  # noqa

    def __str__(self):
        return f"AggregatingSLNet(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size}, lr={self.lr})"  # noqa

    def forward(self, batch):
        encoded = self.encoder(batch)
        dist = self.output_layer(encoded)
        return {"distribution": dist}

    @property
    def optimizer(self):
        if self._opt is None:
            self._opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return self._opt

    def compute_loss(self, output, _input):
        # return self._loss_fn(_input['y'], output["distribution"])
        return self._loss_fn(output["distribution"], _input['y'])
