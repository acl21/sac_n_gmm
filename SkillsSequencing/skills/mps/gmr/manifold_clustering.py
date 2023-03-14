import numpy as np

from pymanopt.manifolds import Product

from SkillsSequencing.skills.mps.gmr.statistics import multivariate_normal
from SkillsSequencing.skills.mps.gmr.manifold_statistics import compute_frechet_mean, compute_weighted_frechet_mean


def manifold_k_means(manifold, data, nb_clusters, initial_means=None, nb_iter_max=100):
    nb_data = data.shape[0]
    # If no initialization provided, initialize randomly
    if initial_means is None:
        initial_means = []
        for k in range(nb_clusters):
            initial_means.append(data[np.random.randint(0, nb_data-1)])
        initial_means = np.array(initial_means)

    # Lloyds algorithm
    cluster_means = np.copy(initial_means)
    cluster_assignments = np.zeros(nb_data)
    previous_assignments = np.ones(nb_data)
    nb_iter = 0

    while np.sum(np.abs((cluster_assignments - previous_assignments))) > 0 and nb_iter < nb_iter_max:
        previous_assignments = cluster_assignments

        # Compute distances to means
        distances_to_means = np.zeros((nb_data, nb_clusters))
        for n in range(nb_data):
            for k in range(nb_clusters):
                distances_to_means[n, k] = manifold.dist(cluster_means[k], data[n])

        # Assign data point to clusters
        cluster_assignments = np.argmin(distances_to_means, axis=1)

        # Update the means
        for k in range(nb_clusters):
            cluster_means[k] = compute_frechet_mean(manifold, data[cluster_assignments == k])

        # Update number of iterations
        nb_iter += 1

    return cluster_means, cluster_assignments


def manifold_gmm_em(manifold, data, nb_states, initial_means=None, initial_covariances=None, initial_priors=None,
                    nb_iter_max=200, max_diff_ll=1e-5, regularization_factor=1e-10, logger=None):
    """
    EM algorithm for a GMM on a Riemannian manifold

    Parameters
    ----------
    :param manifold: Riemannian manifold on which the outputs are living    (pymanopt)
    :param data: data points                                                (array nb_data x data_dimension)
    :param nb_states: number of clusters
    :param initial_means: initial GMM means                                 (array nb_states x data_dimension)
    :param initial_covariances: initial GMM covariance                      (array nb_states x data_dimension x data_dimension)
    :param initial_priors: initial GMM priors                               (array nb_states)

    Optional parameters
    -------------------
    :param nb_iter_max: max number of iterations for EM
    :param max_diff_ll: threshold of log likelihood change for convergence
    :param regularization_factor: regularization for the covariance matrices
    :param logger: logger object optionally provided by the calling function

    Returns
    ----------
    :return: GMM means, covariances, priors, and assignments
    """

    nb_min_steps = 5  # min num iterations

    nb_data = data.shape[0]
    nb_dim = manifold.dim
    means = np.copy(initial_means)
    covariances = np.copy(initial_covariances)
    priors = np.copy(initial_priors)
    LL = np.zeros(nb_iter_max)

    # xts = np.zeros((nb_states, nb_data, nb_dim))
    xts = np.repeat(np.expand_dims(np.copy(data), 0), nb_states, 0)
    for k in range(nb_states):
        for n in range(nb_data):
            xts[k, n, :] = manifold.log(means[k], data[n])

    for it in range(nb_iter_max):
        # print(it)

        # E - step
        L = np.zeros((nb_data, nb_states))
        L_log = np.zeros((nb_data, nb_states))

        for k in range(nb_states):
            if isinstance(manifold, Product):
                # Reshape for multivariate normal function (for product of manifold case)
                xts_k = np.concatenate(np.concatenate(xts[k])).reshape(nb_data, -1)
            else:
                xts_k = xts[k]
            L_log[:, k] = np.log(priors[k]) + multivariate_normal(xts_k, np.zeros_like(xts_k), covariances[k], log=True)

        L = np.exp(L_log)
        GAMMA = L / np.sum(L, axis=1)[:, None]
        GAMMA2 = GAMMA / (np.sum(GAMMA, axis=0) + 1e-10)

        # M-step
        for k in range(nb_states):
            # Update means
            means[k] = compute_weighted_frechet_mean(manifold, data, GAMMA2[:, k])
            for n in range(nb_data):
                xts[k, n, :] = manifold.log(means[k], data[n])

            if isinstance(manifold, Product):
                # Reshape for multivariate normal function (for product of manifold case)
                xts_k = np.concatenate(np.concatenate(xts[k])).reshape(nb_data, -1)
            else:
                xts_k = xts[k]

            # Update covariances
            covariances[k] = np.dot(xts_k.T, np.dot(np.diag(GAMMA2[:, k]), xts_k)) + np.eye(xts_k.shape[1]) * regularization_factor

        # Update priors
        priors = np.mean(GAMMA, axis=0)

        LL[it] = np.mean(np.log(np.sum(L, axis=1)))

        # Check for convergence
        if it > nb_min_steps:
            if LL[it] - LL[it - 1] < max_diff_ll:
                if logger == None:
                    print('Converged after %d iterations: %.3e' % (it, LL[it]))
                else:
                    logger.info('Converged after %d iterations: %.3e' % (it, LL[it]))

                return means, covariances, priors, GAMMA

    if logger == None:
        print(f"GMM did not converge before reaching the maximum {nb_iter_max} number of iterations.")
    else:
        logger.info(f"GMM did not converge before reaching the maximum {nb_iter_max} number of iterations.")
    return means, covariances, priors, GAMMA


def compute_gmm_density(manifold, data, nb_states, means, covariances, priors, log=False):
    """
    Computation of the density for a GMM on a Riemannian manifold.

    Parameters
    ----------
    :param manifold: Riemannian manifold on which the outputs are living    (pymanopt)
    :param data: data points                                                (array nb_data x data_dimension)
    :param nb_states: number of clusters
    :param means:  GMM means                                                (array nb_states x data_dimension)
    :param covariances: GMM covariance                                      (array nb_states x data_dimension x data_dimension)
    :param priors: GMM priors                                               (array nb_states)

    Optional parameters
    -------------------
    :param log: if true, return the log likelihood

    Returns
    ----------
    :return: GMM likelihood for each datapoint
    """
    nb_data = data.shape[0]
    nb_dim = data.shape[1]

    xts = np.zeros((nb_states, nb_data, nb_dim))
    for k in range(nb_states):
        for n in range(nb_data):
            xts[k, n, :] = manifold.log(means[k], data[n])

    states_likelihood = np.zeros((nb_data, nb_states))
    for k in range(nb_states):
        states_likelihood[:, k] = priors[k] * multivariate_normal(xts[k], np.zeros_like(means[k]), covariances[k],
                                                                  log=False)

    likelihood = np.sum(states_likelihood, axis=1)

    if log:
        return np.log(likelihood)
    else:
        return likelihood
