import torch.nn as nn

from .distributions import SLBeta, SLDirichlet


class SLELayer(nn.Module):

    def __init__(self, insize, outsize):
        super().__init__()
        self.insize = insize
        self.outsize = outsize
        self.params_net = nn.Sequential(
            nn.Linear(self.insize, self.outsize + 1),
            nn.Softmax(dim=-1))
        if outsize in [1, 2]:
            self.target_cls = SLBeta
        else:
            self.target_cls = SLDirichlet

    def __repr__(self):
        return f"SLELayer({self.insize}, {self.outsize})"

    def __str__(self):
        return f"SLELayer({self.insize}, {self.outsize})"

    def forward(self, inputs):
        params = self.params_net(inputs)
        beliefs, uncs = params.tensor_split([self.outsize], dim=-1)
        if self.target_cls == SLBeta:
            b, d = beliefs.tensor_split(2, dim=-1)
            args = [b.flatten(), d.flatten(), uncs.flatten()]
        else:
            args = [beliefs, uncs]
        return self.target_cls(*args)
