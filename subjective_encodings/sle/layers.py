import torch
from torch import nn

from .distributions import SLBeta, SLDirichlet


class BetaLayer(nn.Module):

    def __init__(self, insize, threshold=0.5):
        super().__init__()
        self.insize = insize
        self.threshold = threshold
        self.unet = nn.Sequential(
                nn.Linear(self.insize, 1),
                nn.Sigmoid())
        self.bnet = nn.Sequential(
                nn.Linear(self.insize, 1),
                nn.Sigmoid())
        self.params_net = nn.Sequential(
                nn.Linear(self.insize, 3),
                nn.Softmax(dim=-1))

    def __repr__(self):
        return f"BetaLayer({self.insize})"

    def __str__(self):
        return f"BetaLayer({self.insize})"

    def old_forward(self, inputs):
        unc = self.unet(inputs)
        bprime = (self.bnet(inputs) > torch.tensor(self.threshold)).float()
        belief = bprime * (1. - unc)
        disbelief = 1. - (belief + unc)
        return SLBeta(belief, disbelief, unc)

    def forward(self, inputs):
        params = self.params_net(inputs)
        beliefs, disbeliefs, uncs = params.chunk(3, dim=-1)
        return SLBeta(beliefs.squeeze(-1),
                      disbeliefs.squeeze(-1),
                      uncs.squeeze(-1))


class DirichletLayer(nn.Module):

    def __init__(self, insize, outsize):
        super().__init__()
        self.insize = insize
        self.outsize = outsize
        self.params_net = nn.Sequential(
                nn.Linear(self.insize, self.outsize + 1),
                nn.Softmax(dim=-1))

    def __repr__(self):
        return f"DirichletLayer({self.insize}, {self.outsize})"

    def __str__(self):
        return f"DirichletLayer({self.insize}, {self.outsize})"

    def forward(self, inputs):
        params = self.params_net(inputs)
        beliefs, uncs = params.tensor_split([self.outsize], dim=-1)
        return SLDirichlet(beliefs, uncs)
