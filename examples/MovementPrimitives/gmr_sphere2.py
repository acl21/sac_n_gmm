import os
import sys
import types
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

from SkillsSequencing.skills.mps.gmr.manifold_statistics import compute_frechet_mean, compute_weighted_frechet_mean
from SkillsSequencing.skills.mps.gmr.manifold_clustering import manifold_k_means, manifold_gmm_em
from SkillsSequencing.skills.mps.gmr.manifold_gmr import manifold_gmr
from SkillsSequencing.utils.plot_sphere_mayavi import plot_sphere, plot_gaussian_mesh_on_tangent_plane
from SkillsSequencing.utils.sphere_utils import sphere_parallel_transport


if __name__ == '__main__':
    np.random.seed(123445)

    # Load data
    letter = 'C'
    exp_dir = './'
    datapath = '2Dletters/'
    data = loadmat(exp_dir + datapath + '%s.mat' % letter)
    demos = [d['pos'][0][0].T for d in data['demos'][0]]

    # Place all points on a sphere
    demos = np.array(demos)
    demos = 2 * demos / (np.max(demos, axis=1) - np.min(demos, axis=1))[:, None, :]
    demos = np.concatenate((demos, np.ones((demos.shape[0], demos.shape[1], 1))), axis=2)
    demos = demos / np.linalg.norm(demos, axis=2)[:, :, None]

    # Create input and output data
    # Input: Current position, Output: Next position
    demos2 = demos[:, :-1, :]
    target = demos[:, 1:, :]

    # Stack the data
    nb_data = demos2.shape[1]
    nb_data_sup = 50
    nb_samples = 5

    X = demos2[0]
    for i in range(1, nb_samples):
        X = np.vstack([X, demos2[i]])
    Y = target[0]
    for i in range(1, nb_samples):
        Y = np.vstack([Y, target[i]])
    
    # Visualize data (input and output share the same points)
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(700, 700))
    fig = mlab.gcf()
    mlab.clf()
    plot_sphere(figure=fig)
    # Plot data on the sphere
    for p in range(nb_samples):
            mlab.points3d(X[p * nb_data:(p + 1) * nb_data, 0],
                          X[p * nb_data:(p + 1) * nb_data, 1],
                          X[p * nb_data:(p + 1) * nb_data, 2],
                          color=(0.5, 0., 0.5),
                          scale_factor=0.02)
    mlab.view(30, 30)
    mlab.show()

    # Data in format compatible with pymanopt product of manifolds
    data = np.empty((X.shape[0], 2), dtype=object)
    for n in range(nb_data*nb_samples):
        data[n] = [X[n], Y[n]]

    data = np.array(data, dtype=object)
    
    # Create the manifold
    input_dim = 3
    output_dim = 3
    # Input manifold
    input_manifold = Sphere(input_dim)
    # Replace transport operation by parallel transport
    input_manifold.transport = types.MethodType(sphere_parallel_transport, input_manifold)
    # Output manifold
    output_manifold = Sphere(output_dim)
    # Replace transport operation by parallel transport
    output_manifold.transport = types.MethodType(sphere_parallel_transport, output_manifold)
    # Product of manifolds
    manifold = Product([input_manifold, output_manifold])

    # Number of clusters
    nb_clusters = 3

    # If model is saved, load it
    filename = exp_dir + '/gmm_sphere2.npz'
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
    # Start with a first point of a trajectory from the dataset and sample and keep track of the output from the learned GMM
    rand_idx = 0
    x0 = np.copy(X[rand_idx, :])
    new_x = x0
    sampled_path = [new_x]
    dt = 1
    for i in range(199):
        x_dot, sigma_gmr, H = manifold_gmr(new_x.reshape(1, -1), manifold, gmm_means, gmm_covariances, gmm_priors)
        new_x = x_dot[0]
        sampled_path.append(new_x)
    sampled_path = np.array(sampled_path)
    

    # Visualize learned GMM means and covariances (input and outuput states), sampled points (Green)
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(700, 700))
    fig = mlab.gcf()
    mlab.clf()
    plot_sphere(figure=fig)
    
    # Plot target data on the sphere
    for p in range(nb_samples):
            mlab.points3d(Y[p * nb_data:(p + 1) * nb_data, 0],
                          Y[p * nb_data:(p + 1) * nb_data, 1],
                          Y[p * nb_data:(p + 1) * nb_data, 2],
                          color=(0.5, 0., 0.5),
                          scale_factor=0.02)

    # Plot sampled points
    for p in range(len(sampled_path)):
        mlab.points3d(sampled_path[p, 0], sampled_path[p, 1], sampled_path[p, 2],
                      color=(0., 0.6, 0.),
                      scale_factor=0.03)

    # Plot Input (Red) and Output (Green) Gaussians
    for k in range(nb_clusters):
        plot_gaussian_mesh_on_tangent_plane(gmm_means[k, 0], gmm_covariances[k, :3, :3], color=(1, 0, 0))
    
    for k in range(nb_clusters):
        plot_gaussian_mesh_on_tangent_plane(gmm_means[k, 1], gmm_covariances[k, 3:, 3:], color=(0, 1, 0))


    mlab.view(30, 120)
    mlab.show()
