import numpy as np

from SkillsSequencing.skills.mps.gmr.statistics import multivariate_normal


def manifold_gmr(input_data, manifold, gmm_means, gmm_covariances, gmm_priors, nbiter_mu=10,
                 regularization_factor=1e-10):

    nb_data = input_data.shape[0]
    nb_states = gmm_means.shape[0]
    H = np.zeros((nb_states, nb_data))

    # in_idx = list(range(0, manifold.manifolds[0].dim))
    # out_idx = list(range(manifold.manifolds[0].dim, manifold.manifolds[0].dim + manifold.manifolds[1].dim))
    # Changed to below as manifold.dim=d-1 for sphere
    in_idx = list(range(0, gmm_means[0][0].shape[0]))
    out_idx = list(range(gmm_means[0][0].shape[0], gmm_means[0][0].shape[0] + gmm_means[0][1].shape[0]))
    nb_dim = len(in_idx) + len(out_idx)

    # Compute weights
    for n in range(nb_data):
        for k in range(nb_states):
            H[k, n] = gmm_priors[k] * multivariate_normal(manifold.manifolds[0].log(gmm_means[k, 0], input_data[n]),
                                                       np.zeros_like(input_data[n]),
                                                       gmm_covariances[k][in_idx][:, in_idx], log=False)
    H = H / np.sum(H, 0)

    # Eigendecomposition of the covariances for parallel transport
    gmm_covariances_eigenvalues = []
    gmm_covariances_eigenvectors = []

    for k in range(nb_states):
        eigenvalues, eigenvectors = np.linalg.eig(gmm_covariances[k])
        gmm_covariances_eigenvalues.append(eigenvalues)
        gmm_covariances_eigenvectors.append(eigenvectors)

    # Compute estimated mean and covariance for each data
    estimated_outputs = np.zeros((nb_data, len(out_idx)))
    estimated_covariances = np.zeros((nb_data, len(out_idx), len(out_idx)))

    for n in range(nb_data):
        input_point = input_data[n]
        exp_data = gmm_means[np.argmax(H[:, n])][1]

        for it in range(nbiter_mu):
            # print it
            exp_u = np.zeros(len(out_idx))
            trsp_sigma = [np.zeros((nb_dim, nb_dim))] * nb_states
            u_out = np.zeros((len(out_idx), nb_states))
            for k in range(nb_states):
                # Transportation of covariance from mean to expected output
                # Parallel transport of the eigenvectors weighted by the square root of the eigenvalues
                trsp_eigvec = np.zeros_like(gmm_covariances_eigenvectors[k])
                for j in range(gmm_covariances_eigenvectors[k].shape[1]):
                    # Create vectors for product of manifold
                    eigvec = [gmm_covariances_eigenvectors[k][in_idx, j],
                              gmm_covariances_eigenvectors[k][out_idx, j]]
                    transport_to = [input_point, exp_data]

                    # Transport
                    trsp_eigvec_j = manifold.transport(gmm_means[k], transport_to, eigvec)
                    trsp_eigvec[:, j] = np.concatenate(trsp_eigvec_j) * gmm_covariances_eigenvalues[k][j] ** 0.5

                # Reconstruction of parallel transported covariance from eigenvectors
                trsp_sigma[k] = np.dot(trsp_eigvec, trsp_eigvec.T)

                # Gaussian conditioning on tangent space
                trsp_sigma_in = trsp_sigma[k][in_idx][:, in_idx]
                trsp_sigma_out_in = trsp_sigma[k][out_idx][:, in_idx]
                if trsp_sigma_in.ndim == 1:
                    trsp_sigma_in = trsp_sigma_in[:, None]
                    trsp_sigma_out_in = trsp_sigma_out_in[:, None]

                u_out[:, k] = manifold.manifolds[1].log(exp_data, gmm_means[k, 1]) + \
                              np.dot(trsp_sigma_out_in,
                                     np.dot(np.linalg.inv(trsp_sigma_in),
                                            manifold.manifolds[0].log(gmm_means[k, 0], input_point)))

                exp_u += u_out[:, k] * H[k, n]

            # Compute expected mean
            exp_data = manifold.manifolds[1].exp(exp_data, exp_u)

        # Compute expected covariance
        exp_cov = np.zeros((len(out_idx), len(out_idx)))
        for k in range(nb_states):
            trsp_sigma_in = trsp_sigma[k][in_idx][:, in_idx]
            trsp_sigma_out_in = trsp_sigma[k][out_idx][:, in_idx]
            if trsp_sigma_in.ndim == 1:
                trsp_sigma_in = trsp_sigma_in[:, None]
                trsp_sigma_out_in = trsp_sigma_out_in[:, None]

            sigma_tmp = trsp_sigma[k][out_idx][:, out_idx] - \
                        np.dot(trsp_sigma_out_in, np.dot(np.linalg.inv(trsp_sigma_in), trsp_sigma_out_in.T))

            exp_cov += H[k, n] * (sigma_tmp + np.dot(u_out[:, k][:, None], u_out[:, k][None]))

        exp_cov += - np.dot(exp_u[:, None], exp_u[None]) + np.eye(len(out_idx)) * regularization_factor

        estimated_outputs[n] = exp_data
        estimated_covariances[n] = exp_cov

    return estimated_outputs, estimated_covariances, H