import math
import torch
import torch.nn.functional as F
from torch.distributions import (
    Normal,
    Independent,
    Distribution,
)


def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5 * torch.log(one_plus_x / one_minus_x)


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """

    def __init__(self, normal_mean, normal_std):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Independent(Normal(normal_mean, normal_std), 1)

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample((n,))

        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def _log_prob_from_pre_tanh(self, pre_tanh_value):
        """
        Adapted from
        https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73
        correction term is mathematically equivalent to - log(1 - tanh(x)^2).
        """
        log_prob = self.normal.log_prob(pre_tanh_value)
        correction = -2 * (
            math.log(2) - pre_tanh_value - F.softplus(-2 * pre_tanh_value)
        ).sum(dim=-1)
        return (log_prob + correction).unsqueeze(-1)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            # errors or instability at values near 1
            value = torch.clamp(value, -0.999999, 0.999999)
            pre_tanh_value = atanh(value)
        return self._log_prob_from_pre_tanh(pre_tanh_value)

    def rsample_with_pretanh(self):
        """
        Sampling in the reparameterization case.
        """
        z = self.normal.rsample()
        z.requires_grad_()
        return torch.tanh(z), z

    def rsample(self):
        """
        Sampling in the reparameterization case.
        """
        value, pre_tanh_value = self.rsample_with_pretanh()
        return value

    def sample_and_logprob(self):
        pre_tanh_value = self.normal.sample()
        value = torch.tanh(pre_tanh_value)
        value, pre_tanh_value = value.detach(), pre_tanh_value.detach()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p

    def rsample_and_logprob(self):
        value, pre_tanh_value = self.rsample_with_pretanh()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p

    def rsample_logprob_and_pretanh(self):
        value, pre_tanh_value = self.rsample_with_pretanh()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p, pre_tanh_value

    @property
    def mean(self):
        return torch.tanh(self.normal_mean)

    @property
    def stddev(self):
        return self.normal_std
