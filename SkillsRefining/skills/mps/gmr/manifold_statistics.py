import numpy as np
import scipy as sc


def compute_frechet_mean(manifold, data, nb_iter_max=10, convergence_threshold=1e-6):
    nb_data = data.shape[0]

    # Initialize to the first data point
    mean = data[0]
    data_on_tangent_space = np.copy(data)
    distance_previous_mean = np.inf
    nb_iter = 0

    while distance_previous_mean > convergence_threshold and nb_iter < nb_iter_max:
        previous_mean = mean
        for n in range(nb_data):
            data_on_tangent_space[n] = manifold.log(mean, data[n])
        mean = manifold.exp(mean, np.mean(data_on_tangent_space, axis=0))
        distance_previous_mean = manifold.dist(mean, previous_mean)
        nb_iter += 1

    return mean


def compute_weighted_frechet_mean(
    manifold, data, weights, nb_iter_max=10, convergence_threshold=1e-6
):
    nb_data = data.shape[0]

    # Initialize to the first data point
    mean = data[0]
    data_on_tangent_space = np.zeros_like(data)
    distance_previous_mean = np.inf
    nb_iter = 0

    while distance_previous_mean > convergence_threshold and nb_iter < nb_iter_max:
        previous_mean = mean
        for n in range(nb_data):
            data_on_tangent_space[n] = manifold.log(mean, data[n])
        # Weighted mean on the tangent space
        weighted_data_on_tangent_space = weights[:, None] * data_on_tangent_space
        weighted_mean_on_tangent_space = np.sum(weighted_data_on_tangent_space, axis=0)
        # Projection of mean on the tangent space to the manifold
        mean = manifold.exp(mean, weighted_mean_on_tangent_space)
        distance_previous_mean = manifold.dist(mean, previous_mean)
        nb_iter += 1

    return mean


def sample_from_gaussian_on_manifold(manifold, mean, covariance, nb_samples):
    # Generate samples in the tangent space of the mean
    samples = np.dot(
        sc.linalg.sqrtm(covariance), np.random.randn(mean.shape[0], nb_samples)
    )

    # Ensure that they are in the tangent space of the mean
    for n in range(nb_samples):
        samples[:, n] = manifold.projection(mean, samples[:, n])
    # Project on the manifold
    samples_on_manifold = np.zeros_like(samples)
    for n in range(nb_samples):
        samples_on_manifold[:, n] = manifold.exp(mean, samples[:, n])

    return samples_on_manifold
