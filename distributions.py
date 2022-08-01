import torch
import torch.distributions as D


EPS = 1e-18


class SLBeta(D.Beta):
    """
    SLBeta(b, d, u, a=0.5, W=2)

    Beta distribution reparameterized to used belief, disbelief,
    and uncertainty parameters in accordance with Subjective Logic.

    b + d + u must equal 1

    Args:
        b (Tensor): belief parameters.
        d (Tensor): disbelief parameters.
        u (Tensor): uncertainty parameters.
        a (None or Tensor): prior probabilities (default Uniform)
        W (int): prior weight (default 2)

    Example:

        >>> params = torch.softmax(torch.randn(3, 3, 1), dim=1)  # 3 examples and 3 parameters  # noqa
        >>> b = params[:, 0]
        >>> d = params[:, 1]
        >>> u = params[:, 2]
        >>> d = SLBeta(b, d, u)

    See help(torch.distribution.Beta) for more information.
    """

    def __init__(self, b, d, u, a=0.5, W=2):
        # First, convert to tensors
        b = torch.as_tensor(b)
        d = torch.as_tensor(d)
        u = torch.as_tensor(u)
        a = torch.as_tensor(a)
        W = torch.as_tensor(W)

        total = b + d + u
        assert torch.isclose(total, torch.ones_like(total)).all()
        self.b = b
        self.d = d
        self.u = torch.where(u == 0., torch.tensor(EPS), u)
        self.a = a
        self.W = W
        c0, c1 = self.reparameterize()
        super().__init__(c0, c1)

    def __repr__(self):
        return "SLBeta()"

    def __str__(self):
        return "SLBeta()"

    def parameters(self):
        return {'b': self.b, 'd': self.d, 'u': self.u}

    def reparameterize(self):
        b = self.b
        d = self.d
        u = self.u
        a = self.a
        W = self.W
        c0 = ((W * b) / u) + (W * a)
        c1 = ((W * d) / u) + (W * (1 - a))
        return (c0, c1)

    def max_uncertainty(self):
        """
        Uncertainty maximization operation.

        SLT Book Eq. 3.27: the uncertainty maximum is
          \\hat{u}_x = min_i[ p(x_i) / a(x_i) ]

        Then solve P = b + au for b, given we know P, a, and u.
        """
        # Projected probabilities
        # p = self.b + self.a * self.u
        p = self.mean

        new_b = torch.zeros_like(self.b)
        new_d = torch.zeros_like(self.d)
        new_u = torch.zeros_like(self.u)

        neg_idxs = p < 0.5
        new_u[neg_idxs] = (p / self.a)[neg_idxs]
        new_d[neg_idxs] = 1. - new_u[neg_idxs]
        new_b[neg_idxs] = torch.tensor(0.)

        pos_idxs = ~neg_idxs
        new_u[pos_idxs] = ((1. - p) / (1. - self.a))[pos_idxs]
        new_b[pos_idxs] = 1. - new_u[pos_idxs]
        new_d[pos_idxs] = torch.tensor(0.)
        return self.__class__(new_b, new_d, new_u, a=self.a, W=self.W)

    def cumulative_fusion(self, other):
        """
        Aleatory Cumulative fusion operator. See SLT Book 12.3
        For Epistemic Cumulative fusion do

        Beta1.cumulative_fusion(Beta2).max_uncertainty()
        """
        b_num = (self.b * other.u) + (other.b * self.u)
        b_denom = (self.u + other.u) - (self.u * other.u)
        b = b_num / b_denom

        u_num = self.u * other.u
        u_denom = b_denom
        u = u_num / u_denom

        d = 1 - (b + u)

        a_num1 = (self.a * other.u) + (other.a * self.u)
        a_num2 = (self.a + other.a) * (self.u * other.u)
        a_denom = (self.u + other.u) - (2 * self.u * other.u)
        a = (a_num1 - a_num2) / a_denom

        new_dist = self.__class__(b, d, u, a=a, W=self.W)
        return new_dist


class SLDirichlet(D.Dirichlet):
    """
    SLDirichlet(b, u, a=None, W=2)

    Dirichlet distribution reparameterized to used belief and uncertainty
    parameters in accordance with Subjective Logic.

    b.sum(dim=1) + u must equal 1.

    Args:
        b (Tensor): belief parameters.
        u (Tensor): uncertainty parameters.
        a (None or Tensor): prior probabilities (default Uniform)
        W (int): prior weight (default 2)

    Example:

        >>> params = torch.softmax(torch.randn(2, 4, 1), dim=1)  # 2 examples with 3 labels + uncertainty  # noqa
        >>> b = params[:, :2]
        >>> u = params[:, 3]
        >>> d = SLDirichlet(b, u)

    See help(torch.distribution.Dirichlet) for more information.
    """

    def __init__(self, b, u, a=None, W=2):
        b = torch.as_tensor(b)
        assert b.size(1) > 1
        u = torch.as_tensor(u)
        assert u.shape == torch.Size([b.size(0), 1])
        W = torch.as_tensor(W)
        total = b.sum(dim=1, keepdim=True) + u
        assert torch.isclose(total, torch.ones_like(total)).all()

        self.b = b
        self.u = torch.where(u == 0., torch.tensor(EPS), u)

        # If prior not specified, use Uniform
        if a is None:
            a = torch.zeros_like(b).fill_(1 / self.b.size(1))
        self.a = a
        assert self.a.shape == self.b.shape
        self.W = W
        alphas = self.reparameterize()
        super().__init__(alphas)

    def __repr__(self):
        return "SLDirichlet()"

    def __str__(self):
        return "SLDirichlet()"

    def parameters(self):
        return {'b': self.b, 'u': self.u}

    def reparameterize(self):
        alphas = ((self.W * self.b) / self.u) + (self.a * self.W)
        return alphas

    def max_uncertainty(self):
        """
        Uncertainty maximization operation.

        SLT Book Eq. 3.27: the uncertainty maximum is
          \\hat{u}_x = min_i[ p(x_i) / a(x_i) ]

        Then solve P = b + au for b, given we know P, a, and u.
        """
        ps = self.mean
        us = ps / self.a
        vals, idxs = us.sort(dim=1)
        new_u = vals[:, 0:1]  # slicing keeps the dims the same
        new_b = ps - self.a * new_u
        return self.__class__(new_b, new_u, a=self.a, W=self.W)

    def cumulative_fusion(self, other):
        """
        Aleatory Cumulative fusion operator. See SLT Book 12.3
        For Epistemic Cumulative fusion do

        Beta1.cumulative_fusion(Beta2).max_uncertainty()
        """
        b_num = (self.b * other.u) + (other.b * self.u)
        b_denom = (self.u + other.u) - (self.u * other.u)
        b = b_num / b_denom

        u_num = self.u * other.u
        u_denom = b_denom
        u = u_num / u_denom

        a_num1 = (self.a * other.u) + (other.a * self.u)
        a_num2 = (self.a + other.a) * (self.u * other.u)
        a_denom = (self.u + other.u) - (2 * self.u * other.u)
        a = (a_num1 - a_num2) / a_denom

        new_dist = self.__class__(b, u, a=a, W=self.W)
        return new_dist


def lnB(a, b):
    return (torch.lgamma(a) + torch.lgamma(b)) - torch.lgamma(a + b)


def beta_cross_entropy(beta1: SLBeta, beta2: SLBeta):
    """
    Cross entropy between two beta distributions

    H(X1, X2) = \\int -f(x;a,b) ln(f(x;a',b')) dx
         = ln(B(a',b')) - (a'-1)\\psi(a) - (b'-1)\\psi(b) + (a'+b'-2)\\psi(a+b)
    """
    a1 = beta1.concentration0
    b1 = beta1.concentration1
    a2 = beta2.concentration0
    b2 = beta2.concentration1
    adigamma = (a2 - 1.) * torch.digamma(a1)
    bdigamma = (b2 - 1.) * torch.digamma(b1)
    both_digamma = (a2 + b2 - 2) * torch.digamma(a1 + b1)
    ce = lnB(a2, b2) - adigamma - bdigamma + both_digamma
    return ce


def dirichlet_cross_entropy(dir1: SLDirichlet, dir2: SLDirichlet):
    """
    Cross entropy between two dirichlet distributions
    """
    raise NotImplementedError()
