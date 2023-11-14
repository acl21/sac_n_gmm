from gmr.utils import check_random_state
import numpy as np
from .batch_mvn import BatchMVN
import torch


class BatchGMM(object):
    def __init__(
        self,
        n_components,
        priors=None,
        means=None,
        covariances=None,
        verbose=0,
        random_state=None,
    ):
        self.n_components = n_components
        if torch.is_tensor(priors):
            self.priors = priors
        else:
            self.priors = torch.from_numpy(priors)
        if torch.is_tensor(means):
            self.means = means
        else:
            self.means = torch.from_numpy(means)
        if torch.is_tensor(covariances):
            self.covariances = covariances
        else:
            self.covariances = torch.from_numpy(covariances)
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

        if self.priors is not None:
            self.priors = torch.asarray(self.priors)
        if self.means is not None:
            self.means = torch.asarray(self.means)
        if self.covariances is not None:
            self.covariances = torch.asarray(self.covariances)

    def _check_initialized(self):
        if self.priors is None:
            raise ValueError("Priors have not been initialized")
        if self.means is None:
            raise ValueError("Means have not been initialized")
        if self.covariances is None:
            raise ValueError("Covariances have not been initialized")

    def condition(self, indices, batch_x):
        """Conditional distribution over given indices.

        Parameters
        ----------
        indices : array-like, shape (n_new_features,)
            Indices of dimensions that we want to condition.

        x : array-like, shape (n_new_features,)
            Values of the features that we know.

        Returns
        -------
        conditional : GMM
            Conditional GMM distribution p(Y | X=x).
        """
        self._check_initialized()

        batch_size = batch_x.shape[0]
        indices = np.asarray(indices, dtype=int)
        batch_x = torch.atleast_2d(torch.from_numpy(batch_x))
        n_features = self.means.shape[2] - len(indices)
        means = torch.empty((batch_size, self.n_components, n_features))
        covariances = torch.empty(
            (batch_size, self.n_components, n_features, n_features)
        )

        marginal_norm_factors = torch.empty((batch_size, self.n_components))
        marginal_prior_exponents = torch.empty((batch_size, self.n_components))

        for k in range(self.n_components):
            mvn = BatchMVN(
                mean=self.means[:, k],
                covariance=self.covariances[:, k],
                random_state=self.random_state,
            )
            conditioned = mvn.condition(indices, batch_x)
            means[:, k] = conditioned.mean
            covariances[:, k] = conditioned.covariance

            (
                marginal_norm_factors[:, k],
                marginal_prior_exponents[:, k],
            ) = mvn.marginalize(indices).to_norm_factor_and_exponents(batch_x)

        priors = _safe_probability_density(
            self.priors * marginal_norm_factors, marginal_prior_exponents
        )

        return BatchGMM(
            n_components=self.n_components,
            priors=priors,
            means=means,
            covariances=covariances,
            random_state=self.random_state,
        )

    def random_sample_once(self, prior):
        return self.random_state.choice(self.n_components, size=1, p=prior)

    def one_sample_confidence_region(self, alpha):
        self._check_initialized()
        batch_size = self.means.shape[0]
        mvn_indices = np.apply_along_axis(
            self.random_sample_once, axis=1, arr=self.priors
        ).squeeze()
        return BatchMVN(
            mean=self.means[range(batch_size), mvn_indices],
            covariance=self.covariances[range(batch_size), mvn_indices],
            random_state=self.random_state,
        )._one_sample_confidence_region(alpha=alpha)


def _safe_probability_density(norm_factors, exponents):
    """Compute numerically safe probability densities of a GMM.

    The probability density of individual Gaussians in a GMM can be computed
    from a formula of the form
    q_k(X=x) = p_k(X=x) / sum_l p_l(X=x)
    where p_k(X=x) = c_k * exp(exponent_k) so that
    q_k(X=x) = c_k * exp(exponent_k) / sum_l c_l * exp(exponent_l)
    Instead of using computing this directly, we implement it in a numerically
    more stable version that works better for very small or large exponents
    that would otherwise lead to NaN or division by 0.
    The following expression is mathematically equal for any constant m:
    q_k(X=x) = c_k * exp(exponent_k - m) / sum_l c_l * exp(exponent_l - m),
    where we set m = max_l exponents_l.

    Parameters
    ----------
    norm_factors : array, shape (batch_size, n_components,)
        Normalization factors of individual Gaussians

    exponents : array, shape (batch_size, n_components)
        Exponents of each combination of Gaussian and sample

    Returns
    -------
    p : array, shape (batch_size, n_components)
        Probability density of each sample
    """
    m = torch.max(exponents, axis=1)[0].unsqueeze(-1)
    p = norm_factors * torch.exp(exponents - m)
    p /= torch.sum(p, axis=1).unsqueeze(-1)
    return p
