import os
import types
import numpy as np
from scipy.io import loadmat  # loading data from matlab
from mayavi import mlab
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
from pymanopt.manifolds import Euclidean, Sphere, Product

from SkillsRefining.skills.mps.gmr.manifold_statistics import compute_frechet_mean, compute_weighted_frechet_mean
from SkillsRefining.skills.mps.gmr.manifold_clustering import manifold_k_means, manifold_gmm_em
from SkillsRefining.skills.mps.gmr.manifold_gmr import manifold_gmr
from SkillsRefining.utils.plot_sphere_mayavi import plot_sphere, plot_gaussian_mesh_on_tangent_plane
from SkillsRefining.utils.plot_utils import plot_gaussian_covariance
from SkillsRefining.utils.sphere_utils import sphere_parallel_transport


if __name__ == '__main__':
    np.random.seed(123445)

    # Load data
    letterE = 'C'
    letterS = 'C'
    datapath = './2Dletters/'
    dataE = loadmat(datapath + '%s.mat' % letterE)
    dataS = loadmat(datapath + '%s.mat' % letterS)
    demosE = [d['pos'][0][0].T for d in dataE['demos'][0]]
    demosS = [d['pos'][0][0].T for d in dataS['demos'][0]]

    # Parameters
    nb_data = demosE[0].shape[0]
    nb_data_sup = 5
    nb_samples = 5
    dt = 0.01
    input_dim = 1
    output_dim_euclidean = 2
    output_dim_sphere = 3
    output_dim = output_dim_euclidean + output_dim_sphere
    in_idx = [0]
    out1_idx = [1, 2]
    out2_idx = [3, 4, 5]
    nb_states = 3

    # Data
    X = dt * np.concatenate([np.arange(demosE[i].shape[0])[:, None] for i in range(nb_samples)])
    Y1 = np.concatenate(demosE[:nb_samples])
    Y2 = np.concatenate(demosS[:nb_samples])
    
    # Output on a sphere
    Y2 = 2 * Y2 / (np.max(Y2, axis=0) - np.min(Y2, axis=0))
    Y2 = np.hstack((Y2, np.ones((Y2.shape[0], 1))))
    Y2 = Y2 / np.linalg.norm(Y2, axis=1)[:, None]

    # Data in format compatible with pymanopt product of manifolds
    data = np.empty((nb_data*nb_samples, 3), dtype=object)
    for n in range(nb_data*nb_samples):
        data[n] = [X[n], Y1[n], Y2[n]]

    # Test data
    Xt = dt * np.arange(nb_data + nb_data_sup)[:, None]

    # Create the manifold
    input_manifold = Euclidean(input_dim)
    output_manifold1 = Euclidean(output_dim_euclidean)
    output_manifold2 = Sphere(output_dim_sphere)
    # Replace transport operation by parallel transport
    # output_manifold2.transport = types.MethodType(sphere_parallel_transport, output_manifold2)
    # Create the product of manifold for GMM and GMR
    manifold = Product([input_manifold, output_manifold1, output_manifold2])

    # Number of clusters
    nb_clusters = 3

    # If model is saved, load it
    filename = './gmm_product.npz'
    if os.path.isfile(filename):
        gmm = np.load(filename)
        gmm.allow_pickle = True
        gmm_means = np.array(gmm['gmm_means'])
        gmm_covariances = np.array(gmm['gmm_covariances'])
        gmm_priors = np.array(gmm['gmm_priors'])

    # Otherwise train GMM
    else:
        # K-means
        km_means, km_assignments = manifold_k_means(manifold, data, nb_clusters=nb_clusters)

        # GMM
        initial_covariances = np.concatenate(nb_clusters * [np.eye(input_dim+output_dim)[None]], 0)
        initial_priors = np.zeros(nb_clusters)
        for k in range(nb_clusters):
            initial_priors[k] = np.sum(km_assignments == k) / nb_data
        gmm_means, gmm_covariances, gmm_priors, gmm_assignments = manifold_gmm_em(manifold, data, nb_clusters,
                                                                                  initial_means=km_means,
                                                                                  initial_covariances=initial_covariances,
                                                                                  initial_priors=initial_priors
                                                                                  )
        np.savez(filename, gmm_means=gmm_means, gmm_covariances=gmm_covariances, gmm_priors=gmm_priors)

    # GMR
    mu_gmr, sigma_gmr, H = manifold_gmr(Xt, manifold, gmm_means, gmm_covariances, gmm_priors,
                                        in_manifold_idx=[0], out_manifold_idx=[1, 2])

    # Plots
    # Euclidean part
    # Plot the GMM assignments and means
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    # Plot data on the sphere
    for p in range(nb_samples):
        # plt.plot(data[n, 0], data[n, 1], '.', markersize=12, color='k')
        plt.plot(Y1[p * nb_data:(p + 1) * nb_data, 0], Y1[p * nb_data:(p + 1) * nb_data, 1],
                 '.', markersize=12, color=(0., 0., 0.))

    # Plot GMM means and covariances
    for k in range(nb_clusters):
        plt.plot(gmm_means[k, 1][0], gmm_means[k, 1][1], color=(0.5, 0, 0.2))
        plot_gaussian_covariance(ax, gmm_means[k, 1], gmm_covariances[k][out1_idx][:, out1_idx], color=(0.5, 0, 0.2))

    # Plot GMR
    plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], 'o', markersize=12, color=(0.20, 0.54, 0.93))
    for n in range(nb_data + nb_data_sup):
        plot_gaussian_covariance(ax, mu_gmr[n, 0:2], sigma_gmr[n][:2][:, :2], color=(0.20, 0.54, 0.93), transparency=0.1)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.locator_params(nbins=4)
    plt.tight_layout()
    plt.show()

    # Sphere part
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(700, 700))
    fig = mlab.gcf()
    mlab.clf()
    plot_sphere(figure=fig)
    # Plot data on the sphere
    for p in range(nb_samples):
        mlab.points3d(Y2[p * nb_data:(p + 1) * nb_data, 0],
                      Y2[p * nb_data:(p + 1) * nb_data, 1],
                      Y2[p * nb_data:(p + 1) * nb_data, 2],
                      color=(0., 0., 0.),
                      scale_factor=0.03)

    # Plot Gaussians
    for k in range(nb_clusters):
        plot_gaussian_mesh_on_tangent_plane(gmm_means[k, 2], gmm_covariances[k][out2_idx][:, out2_idx], color=(0.5, 0, 0.2))

    # Plot GMR trajectory
    for n in range(nb_data + nb_data_sup):
        # Plot mean and covariance
        plot_gaussian_mesh_on_tangent_plane(mu_gmr[n, 2:], sigma_gmr[n][2:][:, 2:], color=(0.20, 0.54, 0.93))
        # Plot mean only
        # mlab.points3d(mu_gmr[n, 2], mu_gmr[n, 3], mu_gmr[n, 4],
        #               color=(0.20, 0.54, 0.93),
        #               scale_factor=0.03)
    mlab.view(30, 120)
    mlab.show()

