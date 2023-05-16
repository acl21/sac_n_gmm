import os
import sys
from pathlib import Path

cwd_path = Path(__file__).absolute().parents[0]
parent_path = cwd_path.parents[0]
sys.path.insert(0, parent_path.as_posix())
sys.path.insert(0, cwd_path.parents[0].parents[0].as_posix()) # Root

import numpy as np
from scipy.io import loadmat  # loading data from matlab
from mayavi import mlab
import matplotlib.pyplot as plt
from pymanopt.manifolds import Euclidean, Sphere, Product

from SkillsRefining.skills.mps.gmr.manifold_statistics import compute_frechet_mean, compute_weighted_frechet_mean
from SkillsRefining.skills.mps.gmr.manifold_clustering import manifold_k_means, manifold_gmm_em
from SkillsRefining.skills.mps.gmr.manifold_gmr import manifold_gmr
from SkillsRefining.utils.plot_sphere_mayavi import plot_sphere, plot_gaussian_mesh_on_tangent_plane


if __name__ == '__main__':
    np.random.seed(123445)

    # Load data
    letter = 'C'
    exp_dir = './examples/MovementPrimitives/'
    datapath = '2Dletters/'
    data = loadmat(exp_dir + datapath + '%s.mat' % letter)
    demos = [d['pos'][0][0].T for d in data['demos'][0]]

    # Parameters
    nb_data = demos[0].shape[0]
    nb_data_sup = 50
    nb_samples = 5
    dt = 0.01
    input_dim = 1
    output_dim = 3
    in_idx = [0]
    out_idx = [1, 2, 3]
    nb_states = 3

    # Create time data
    demos_t = [np.arange(demos[i].shape[0])[:, None] for i in range(nb_samples)]

    # Stack time and position data
    demos_tx = [np.hstack([demos_t[i] * dt, demos[i]]) for i in range(nb_samples)]

    # Stack demos
    demos_np = demos_tx[0]
    for i in range(1, nb_samples):
        demos_np = np.vstack([demos_np, demos_tx[i]])

    X = demos_np[:, 0][:, None]
    Y = demos_np[:, 1:]

    # Output on a sphere
    Y = 2 * Y / (np.max(Y, axis=0) - np.min(Y, axis=0))
    Y = np.hstack((Y, np.ones((Y.shape[0], 1))))
    Y = Y / np.linalg.norm(Y, axis=1)[:, None]

    # Data in format compatible with pymanopt product of manifolds
    data = []
    for n in range(nb_data*nb_samples):
        data.append([X[n], Y[n]])
    data = np.array(data)
    # data = demos_np

    # Test data
    Xt = dt * np.arange(nb_data + nb_data_sup)[:, None]

    # Create the manifold
    input_manifold = Euclidean(input_dim)
    output_manifold = Sphere(output_dim)
    manifold = Product([input_manifold, output_manifold])

    # Number of clusters
    nb_clusters = 3

    # If model is saved, load it
    filename = exp_dir + '/gmm_sphere.npz'
    if os.path.isfile(filename) and False:
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
    mu_gmr, sigma_gmr, H = manifold_gmr(Xt, manifold, gmm_means, gmm_covariances, gmm_priors)

    # Plots
    # Plot sphere
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(700, 700))
    fig = mlab.gcf()
    mlab.clf()
    plot_sphere(figure=fig)
    # Plot data on the sphere
    for p in range(nb_samples):
        mlab.points3d(Y[p * nb_data:(p + 1) * nb_data, 0],
                      Y[p * nb_data:(p + 1) * nb_data, 1],
                      Y[p * nb_data:(p + 1) * nb_data, 2],
                      color=(0., 0., 0.),
                      scale_factor=0.03)

    # # Plot Gaussians
    # for k in range(nb_clusters):
    #     plot_gaussian_mesh_on_tangent_plane(gmm_means[k, 1], gmm_covariances[k, 1:, 1:], color=(0.5, 0, 0.2))

    # # # Plot GMR trajectory
    # for n in range(nb_data + nb_data_sup):
    #     # Plot mean and covariance
    #     plot_gaussian_mesh_on_tangent_plane(mu_gmr[n], sigma_gmr[n], color=(0.20, 0.54, 0.93))
    #     # Plot mean only
    #     mlab.points3d(mu_gmr[n, 0], mu_gmr[n, 1], mu_gmr[n, 2],
    #                   color=(0.20, 0.54, 0.93),
    #                   scale_factor=0.03)
    mlab.view(30, 120)
    # mlab.savefig('Figure0.png')
    mlab.show()

    plt.figure(figsize=(5, 4))
    for p in range(nb_samples):
        plt.plot(Xt[:nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 0], color=[.7, .7, .7])
    plt.plot(Xt[:, 0], mu_gmr[:, 0], color=[0.20, 0.54, 0.93], linewidth=3)
    miny = mu_gmr[:, 0] - np.sqrt(sigma_gmr[:, 0, 0])
    maxy = mu_gmr[:, 0] + np.sqrt(sigma_gmr[:, 0, 0])
    plt.fill_between(Xt[:, 0], miny, maxy, color=[0.20, 0.54, 0.93], alpha=0.3)
    axes = plt.gca()
    axes.set_ylim([-1.1, 1.1])
    plt.xlabel('$t$', fontsize=30)
    plt.ylabel('$y_1$', fontsize=30)
    plt.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(exp_dir + 'Figure1.png', dpi=100)

    plt.figure(figsize=(5, 4))
    for p in range(nb_samples):
        plt.plot(Xt[:nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 1], color=[.7, .7, .7])
    plt.plot(Xt[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3)
    miny = mu_gmr[:, 1] - np.sqrt(sigma_gmr[:, 1, 1])
    maxy = mu_gmr[:, 1] + np.sqrt(sigma_gmr[:, 1, 1])
    plt.fill_between(Xt[:, 0], miny, maxy, color=[0.20, 0.54, 0.93], alpha=0.3)
    axes = plt.gca()
    axes.set_ylim([-1.1, 1.1])
    plt.xlabel('$t$', fontsize=30)
    plt.ylabel('$y_2$', fontsize=30)
    plt.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(exp_dir + 'Figure2.png', dpi=100)
    # plt.show()

    plt.figure(figsize=(5, 4))
    for p in range(nb_samples):
        plt.plot(Xt[:nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 2], color=[.7, .7, .7])
    plt.plot(Xt[:, 0], mu_gmr[:, 2], color=[0.20, 0.54, 0.93], linewidth=3)
    miny = mu_gmr[:, 2] - np.sqrt(sigma_gmr[:, 2, 2])
    maxy = mu_gmr[:, 2] + np.sqrt(sigma_gmr[:, 2, 2])
    plt.fill_between(Xt[:, 0], miny, maxy, color=[0.20, 0.54, 0.93], alpha=0.3)
    axes = plt.gca()
    axes.set_ylim([-1.1, 1.1])
    plt.xlabel('$t$', fontsize=30)
    plt.ylabel('$y_3$', fontsize=30)
    plt.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(exp_dir + 'Figure3.png', dpi=100)
    # plt.show()
