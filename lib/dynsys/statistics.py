import numpy as np


def multivariate_normal(x, mu, sigma=None, log=True, inv_sigma=None):
    """
    Multivariatve normal distribution PDF

    :param x:		np.array([nb_samples, nb_dim])
    :param mu: 		np.array([nb_dim])
    :param sigma: 	np.array([nb_dim, nb_dim])
    :param log: 	bool
    :return:
    """
    dx = x - mu
    if sigma.ndim == 1:
        sigma = sigma[:, None]
        dx = dx[:, None]
        inv_sigma = np.linalg.inv(sigma) if inv_sigma is None else inv_sigma
        log_lik = -0.5 * np.sum(np.dot(dx, inv_sigma) * dx, axis=1) - 0.5 * np.log(
            np.linalg.det(2 * np.pi * sigma)
        )
    else:
        inv_sigma = np.linalg.inv(sigma) if inv_sigma is None else inv_sigma
        log_lik = -0.5 * np.einsum(
            "...j,...j", dx, np.einsum("...jk,...j->...k", inv_sigma, dx)
        ) - 0.5 * np.log(np.linalg.det(2 * np.pi * sigma))

    return log_lik if log else np.exp(log_lik)


def pca(centered_data, nb_components):
    covariance = np.cov(centered_data.T, bias=True)
    eigenvals, eigenvecs = np.linalg.eig(covariance)

    idx = eigenvals.argsort()[::-1]
    eigenvals = np.real(eigenvals[idx])
    eigenvecs = np.real(eigenvecs[:, idx])

    return eigenvecs[:, :nb_components], eigenvals[:nb_components]
