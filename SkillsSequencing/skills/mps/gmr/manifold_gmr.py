import numpy as np
from scipy.linalg import block_diag
from pymanopt.manifolds import Euclidean, Sphere, Product

from SkillsSequencing.skills.mps.gmr.statistics import multivariate_normal


def manifold_gmr(input_data, manifold, gmm_means, gmm_covariances, gmm_priors, in_manifold_idx=[0],
                 out_manifold_idx=[1], nbiter_mu=10, regularization_factor=1e-10, convergence_threshold=1e-6):

    nb_data = input_data.shape[0]
    nb_states = len(gmm_means)
    H = np.zeros((nb_states, nb_data))

    nb_dim_in = np.sum([gmm_means[0][i].shape[0] for i in in_manifold_idx])
    nb_dim_out = np.sum([gmm_means[0][i].shape[0] for i in out_manifold_idx])
    in_idx = list(range(0, nb_dim_in))
    out_idx = list(range(nb_dim_in, nb_dim_in + nb_dim_out))
    nb_dim = len(in_idx) + len(out_idx)

    # Input and output manifolds
    if len(in_manifold_idx) > 1:
        in_manifold = Product(manifold.manifolds[slice(in_manifold_idx[0], in_manifold_idx[-1] + 1)])
    else:
        in_manifold_idx = in_manifold_idx[0]
        in_manifold = manifold.manifolds[in_manifold_idx]
    if len(out_manifold_idx) > 1:
        out_manifold = Product(manifold.manifolds[slice(out_manifold_idx[0], out_manifold_idx[-1] + 1)])
    else:
        out_manifold_idx = out_manifold_idx[0]
        out_manifold = manifold.manifolds[out_manifold_idx]

    # Compute weights
    for n in range(nb_data):
        for k in range(nb_states):
            H[k, n] = gmm_priors[k] * multivariate_normal(in_manifold.log(gmm_means[k][in_manifold_idx], input_data[n]),
                                                       np.zeros_like(input_data[n]),
                                                       gmm_covariances[k][in_idx][:, in_idx], log=False)
    H = H / np.sum(H, 0)

    # Eigendecomposition of the covariances for parallel transport
    gmm_covariances_eigenvalues = []
    gmm_covariances_eigenvectors = []
    gmm_covariances_eigenvectors_product = []
    for k in range(nb_states):
        eigenvalues, eigenvectors = np.linalg.eig(gmm_covariances[k])
        gmm_covariances_eigenvalues.append(eigenvalues)
        gmm_covariances_eigenvectors.append(eigenvectors)

    # Eigenvectors in a format supported for the product of manifolds
    for k in range(nb_states):
        input_point = input_data[n]
        exp_data = gmm_means[np.argmax(H[:, n])][out_manifold_idx]
        gmm_covariances_eigenvectors_product_k = []
        for j in range(gmm_covariances_eigenvectors[k].shape[1]):
            # Create vectors for product of manifold
            if isinstance(in_manifold, Product):
                eigvec_in = np.empty(len(in_manifold.manifolds), dtype=object)
                idx = 0
                for m in range(len(in_manifold.manifolds)):
                    eigvec_in[m] = gmm_covariances_eigenvectors[k][in_idx[idx: idx + input_point[m].shape[0]], j]
                    idx += input_point[m].shape[0]
            else:
                eigvec_in = np.empty(1, dtype=object)
                eigvec_in[0] = gmm_covariances_eigenvectors[k][in_idx, j]

            if isinstance(out_manifold, Product):
                eigvec_out = np.empty(len(out_manifold.manifolds), dtype=object)
                idx = 0
                for m in range(len(out_manifold.manifolds)):
                    eigvec_out[m] = gmm_covariances_eigenvectors[k][out_idx[idx: idx + exp_data[m].shape[0]], j]
                    idx += exp_data[m].shape[0]
            else:
                eigvec_out = np.empty(1, dtype=object)
                eigvec_out[0] = gmm_covariances_eigenvectors[k][out_idx, j]

            gmm_covariances_eigenvectors_product_k.append(np.hstack((eigvec_in, eigvec_out)))
        gmm_covariances_eigenvectors_product.append(gmm_covariances_eigenvectors_product_k)

    # Compute estimated mean and covariance for each data
    estimated_outputs = np.zeros((nb_data, len(out_idx)))
    estimated_covariances = np.zeros((nb_data, len(out_idx), len(out_idx)))

    for n in range(nb_data):
        input_point = input_data[n]
        exp_data = gmm_means[np.argmax(H[:, n])][out_manifold_idx]

        # Basis vectors for changes of basis for inverse computation if the input manifold is a sphere
        change_of_basis = False
        if isinstance(in_manifold, Sphere):
            change_of_basis = True
            manifold_origin = np.zeros(len(in_idx))
            manifold_origin[-1] = 1.0
            # We have y_tgt_ambiant = B * y_tgt_2d and y_tgt_2d = B.T * y_tgt_ambiant
            basis_origin = np.eye(len(in_idx))[:, :-1]
            basis_transported = np.zeros_like(basis_origin)
            for j in range(basis_origin.shape[1]):
                basis_transported[:, j] = in_manifold.transport(manifold_origin,
                                                                          input_point,
                                                                          basis_origin[:, j])
        elif isinstance(in_manifold, Product):
            if any([isinstance(in_manifold.manifolds[m], Sphere) for m in range(len(in_manifold.manifolds))]):
                change_of_basis = True
                basis_transported = []
                for m in range(len(in_manifold.manifolds)):
                    if isinstance(in_manifold.manifolds[m], Euclidean):
                        basis_transported.append(np.eye(input_point[m].shape[0]))
                    elif isinstance(in_manifold.manifolds[m], Sphere):
                        manifold_origin = np.zeros(input_point[m].shape[0])
                        manifold_origin[-1] = 1.0
                        # We have y_tgt_ambiant = B * y_tgt_2d and y_tgt_2d = B.T * y_tgt_ambiant
                        basis_origin = np.eye(input_point[m].shape[0])[:, :-1]
                        basis_trsp = np.zeros_like(basis_origin)
                        for j in range(basis_origin.shape[1]):
                            basis_trsp[:, j] = in_manifold.manifolds[m].transport(manifold_origin, input_point[m],
                                                                                  basis_origin[:, j])
                        basis_transported.append(basis_trsp)
                    else:
                        raise NotImplementedError
                basis_transported = block_diag(*basis_transported)

        # Compute expected mean
        distance_previous_mean = np.inf
        nb_iter = 0
        while distance_previous_mean > convergence_threshold and nb_iter < nbiter_mu:
            exp_u = np.zeros(len(out_idx))
            trsp_sigma = [np.zeros((nb_dim, nb_dim))] * nb_states
            u_out = np.zeros((len(out_idx), nb_states))
            for k in range(nb_states):
                # Transportation of covariance from mean to expected output
                # Parallel transport of the eigenvectors weighted by the square root of the eigenvalues
                trsp_eigvec = np.zeros_like(gmm_covariances_eigenvectors[k])
                if not isinstance(in_manifold, Product) and not isinstance(out_manifold, Product):
                    transport_to = [input_point, exp_data]
                else:
                    transport_to = np.hstack((input_point, exp_data))

                for j in range(gmm_covariances_eigenvectors[k].shape[1]):
                    # Transport
                    trsp_eigvec_j = manifold.transport(gmm_means[k], transport_to,
                                                       gmm_covariances_eigenvectors_product[k][j])
                    trsp_eigvec[:, j] = np.hstack(trsp_eigvec_j) * gmm_covariances_eigenvalues[k][j] ** 0.5

                # Reconstruction of parallel transported covariance from eigenvectors
                trsp_sigma[k] = np.dot(trsp_eigvec, trsp_eigvec.T)

                # Gaussian conditioning on tangent space
                trsp_sigma_in = trsp_sigma[k][in_idx][:, in_idx]
                trsp_sigma_out_in = trsp_sigma[k][out_idx][:, in_idx]
                if trsp_sigma_in.ndim == 1:
                    trsp_sigma_in = trsp_sigma_in[:, None]
                    trsp_sigma_out_in = trsp_sigma_out_in[:, None]

                if not change_of_basis:
                    inv_trsp_sigma_in = np.linalg.inv(trsp_sigma_in)
                else:
                    inv_trsp_sigma_in = np.dot(basis_transported,
                                               np.dot(np.linalg.inv(np.dot(basis_transported.T,
                                                                           np.dot(trsp_sigma_in, basis_transported))),
                                                      basis_transported.T))

                if isinstance(in_manifold, Product):
                    log_mean_in = np.concatenate(in_manifold.log(gmm_means[k][in_manifold_idx], input_point))
                else:
                    log_mean_in = in_manifold.log(gmm_means[k][in_manifold_idx], input_point)
                if isinstance(out_manifold, Product):
                    log_mean_out = np.concatenate(out_manifold.log(exp_data, gmm_means[k][out_manifold_idx]))
                else:
                    log_mean_out = out_manifold.log(exp_data, gmm_means[k][out_manifold_idx])

                u_out[:, k] = log_mean_out + np.dot(trsp_sigma_out_in, np.dot(inv_trsp_sigma_in, log_mean_in))
                exp_u += u_out[:, k] * H[k, n]

            # If output manifold is a product, reshape exp_u
            if isinstance(out_manifold, Product):
                exp_u_reshaped = np.empty(len(out_manifold.manifolds), dtype=object)
                idx = 0
                for m in range(len(out_manifold.manifolds)):
                    exp_u_reshaped[m] = exp_u[idx: idx + exp_data[m].shape[0]]
                    idx += exp_data[m].shape[0]
                exp_u = exp_u_reshaped

            # Project mean from the tangent space onto the manifold
            exp_data_old = exp_data
            exp_data = out_manifold.exp(exp_data, exp_u)
            distance_previous_mean = out_manifold.dist(exp_data, exp_data_old)
            nb_iter += 1

        # Compute expected covariance
        exp_cov = np.zeros((len(out_idx), len(out_idx)))
        exp_u = np.zeros(len(out_idx))
        trsp_sigma = [np.zeros((nb_dim, nb_dim))] * nb_states
        u_out = np.zeros((len(out_idx), nb_states))
        for k in range(nb_states):
            # Transportation of covariance from mean to expected output
            # Parallel transport of the eigenvectors weighted by the square root of the eigenvalues
            trsp_eigvec = np.zeros_like(gmm_covariances_eigenvectors[k])
            if not isinstance(in_manifold, Product) and not isinstance(out_manifold, Product):
                transport_to = [input_point, exp_data]
            else:
                transport_to = np.hstack((input_point, exp_data))

            for j in range(gmm_covariances_eigenvectors[k].shape[1]):
                # Transport
                trsp_eigvec_j = manifold.transport(gmm_means[k], transport_to,
                                                   gmm_covariances_eigenvectors_product[k][j])
                trsp_eigvec[:, j] = np.hstack(trsp_eigvec_j) * gmm_covariances_eigenvalues[k][j] ** 0.5

            # Reconstruction of parallel transported covariance from eigenvectors
            trsp_sigma[k] = np.dot(trsp_eigvec, trsp_eigvec.T)

            # Compute the covariance
            trsp_sigma_in = trsp_sigma[k][in_idx][:, in_idx]
            trsp_sigma_out_in = trsp_sigma[k][out_idx][:, in_idx]
            if trsp_sigma_in.ndim == 1:
                trsp_sigma_in = trsp_sigma_in[:, None]
                trsp_sigma_out_in = trsp_sigma_out_in[:, None]

            if not change_of_basis:
                inv_trsp_sigma_in = np.linalg.inv(trsp_sigma_in)
            else:
                inv_trsp_sigma_in = np.dot(basis_transported,
                                           np.dot(np.linalg.inv(np.dot(basis_transported.T,
                                                                       np.dot(trsp_sigma_in, basis_transported))),
                                                  basis_transported.T))
            if isinstance(in_manifold, Product):
                log_mean_in = np.concatenate(in_manifold.log(gmm_means[k][in_manifold_idx], input_point))
            else:
                log_mean_in = in_manifold.log(gmm_means[k][in_manifold_idx], input_point)
            if isinstance(out_manifold, Product):
                log_mean_out = np.concatenate(out_manifold.log(exp_data, gmm_means[k][out_manifold_idx]))
            else:
                log_mean_out = out_manifold.log(exp_data, gmm_means[k][out_manifold_idx])
            u_out[:, k] = log_mean_out + np.dot(trsp_sigma_out_in, np.dot(inv_trsp_sigma_in, log_mean_in))
            exp_u += u_out[:, k] * H[k, n]

            # Covariance part obtained from parallel transported component covariance
            sigma_tmp = trsp_sigma[k][out_idx][:, out_idx] - \
                        np.dot(trsp_sigma_out_in, np.dot(inv_trsp_sigma_in, trsp_sigma_out_in.T))
            # Add component mean contribution
            exp_cov += H[k, n] * (sigma_tmp + np.dot(u_out[:, k][:, None], u_out[:, k][None]))

        # Add expected mean contribution
        exp_cov += - np.dot(exp_u[:, None], exp_u[None]) + np.eye(len(out_idx)) * regularization_factor

        if isinstance(out_manifold, Product):
            exp_data = np.concatenate(exp_data)

        estimated_outputs[n] = exp_data
        estimated_covariances[n] = exp_cov

    return estimated_outputs, estimated_covariances, H

