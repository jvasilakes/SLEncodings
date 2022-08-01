import torch
from torch import nn

from distributions import SLBeta


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

    def __repr__(self):
        return f"BetaLayer({self.insize})"

    def __str__(self):
        return f"BetaLayer({self.insize})"

    def forward(self, inputs):
        unc = self.unet(inputs)
        bprime = (self.bnet(inputs) > torch.tensor(self.threshold)).float()
        belief = bprime * (1. - unc)
        disbelief = 1. - (belief + unc)
        return SLBeta(belief, disbelief, unc).max_uncertainty()


class DirichletLayer(nn.Module):

    def __init__(self, insize, outsize):
        raise NotImplementedError()

    def __repr__(self):
        return f"DirichletLayer({self.insize}, {self.outsize})"

    def __str__(self):
        return f"DirichletLayer({self.insize}, {self.outsize})"

    def forward(self, inputs):
        raise NotImplementedError()
