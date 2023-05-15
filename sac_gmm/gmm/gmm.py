import os
import sys
from pathlib import Path

cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[0]
root = sac_gmm_path.parents[0]

# This is to access the locally installed repo clone when using slurm
sys.path.insert(0, sac_gmm_path.as_posix()) # sac_gmm
sys.path.insert(0, root.as_posix()) # Root

import numpy as np
from pymanopt.manifolds import Euclidean, Sphere, Product

from sac_gmm.gmm.gmr.manifold_clustering import manifold_k_means, manifold_gmm_em
from sac_gmm.gmm.gmr.manifold_gmr import manifold_gmr
from sac_gmm.gmm.utils.plot_utils import visualize_3d_gmm

import logging

class ManifoldGMM(object):
    def __init__(self, n_components=3, plot=False):
        self.n_comp = n_components
        self.plot = plot
        self.name = 'gmm'
        # Data and Manifold
        self.dataset = None
        self.state_type = None
        self.dim = None
        self.manifold = None
        self.data = None
        # GMM
        self.means = None
        self.covariances = None
        self.priors = None
        self.assignments = None
        # Misc
        self.skills_dir = None
        self.logs_outdir = None
        self.logger = logging.getLogger('ManifoldGMM')
        # Start and Goal states
        self.start = None
        self.goal = None

    def make_manifold(self, dim):
        if self.state_type in ['pos', 'joint']:
            in_manifold = Euclidean(dim)
            out_manifold = Euclidean(dim)
        elif self.state_type == 'ori':
            in_manifold = Sphere(dim)
            out_manifold = Sphere(dim)
        elif self.state_type == 'pos_ori':
            manifold = None
        manifold = Product([in_manifold, out_manifold])
        return manifold

    def preprocess_data(self, dataset, normalize=False):
        # Stack position and velocity data
        demos_xdx = [np.hstack([dataset.X[i], dataset.dX[i]]) for i in range(dataset.X.shape[0])]
        # Stack demos
        demos = demos_xdx[0]
        for i in range(1, dataset.X.shape[0]):
            demos = np.vstack([demos, demos_xdx[i]])

        X = demos[:, :self.dim]
        Y = demos[:, self.dim:]

        data = np.empty((X.shape[0], 2), dtype=object)
        for n in range(X.shape[0]):
            data[n] = [X[n], Y[n]]
        return data

    def load_params(self, filename='/gmm_params.npz'):
        self.logger.info(f'Loading GMM params from {self.skills_dir + filename}')
        gmm = np.load(self.skills_dir + filename)
        gmm.allow_pickle = True
        self.means = np.array(gmm['gmm_means'])
        self.covariances = np.array(gmm['gmm_covariances'])
        self.priors = np.array(gmm['gmm_priors'])

    def save_params(self, filename='/gmm_params.npz'):
        np.savez(self.skills_dir + filename, gmm_means=self.means, \
                 gmm_covariances=self.covariances, \
                 gmm_priors=self.priors)
        self.logger.info(f'Saved GMM params at {self.skills_dir + filename}')

    def set_data_params(self, dataset):
        self.dataset = dataset
        self.state_type = self.dataset.state_type
        self.dim = self.dataset.X.numpy().shape[-1]
        self.manifold = self.make_manifold(self.dim)
        self.data = self.preprocess_data(dataset, normalize=False)

    def train(self, dataset):
        # Dataset
        self.set_data_params(dataset)

        # K-Means
        km_means, km_assignments = manifold_k_means(self.manifold, self.data, \
                                                    nb_clusters=self.n_comp)
        # GMM
        self.logger.info(f'Manifold GMM with K-Means priors')
        init_covariances = np.concatenate(self.n_comp * [np.eye(self.dim+self.dim)[None]], 0)
        init_priors = np.zeros(self.n_comp)
        for k in range(self.n_comp):
            init_priors[k] = np.sum(km_assignments == k) / len(km_assignments)
        self.means, self.covariances, self.priors, self.assignments = manifold_gmm_em(self.manifold, self.data, self.n_comp,
                                                                      initial_means=km_means,
                                                                      initial_covariances=init_covariances,
                                                                      initial_priors=init_priors,
                                                                      logger = self.logger)

        # Save GMM params
        self.save_params()

        # Plot GMM
        if self.plot:
            outfile = self.plot_gmm()

    def gmr(self, Xt):
        mu_gmr, sigma_gmr, H = manifold_gmr(Xt, self.manifold, self.means, self.covariances, self.priors)
        return mu_gmr, sigma_gmr, H

    def predict_dx(self, x):
        dx, _, __ = manifold_gmr(x.reshape(1, -1), self.manifold, self.means, self.covariances, self.priors)
        return dx

    def plot_gmm_mlab(self, input_space=True):
        from mayavi import mlab
        from SkillsSequencing.utils.plot_sphere_mayavi import plot_sphere, plot_gaussian_mesh_on_tangent_plane

        if input_space:
            dim = 0
        else:
            dim = 1
        nb_data = self.dataset.X[0].shape[0]
        X = np.concatenate(self.data[:, dim]).reshape(self.data.shape[0], len(self.data[0, 0]))
        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(700, 700))
        fig = mlab.gcf()
        mlab.clf()
        plot_sphere(figure=fig)
        # Plot data on the sphere
        for p in range():
            mlab.points3d(X[p * nb_data:(p + 1) * nb_data, 0],
                          X[p * nb_data:(p + 1) * nb_data, 1],
                          X[p * nb_data:(p + 1) * nb_data, 2],
                          color=(0., 0., 0.),
                          scale_factor=0.03)
        # Plot Gaussians
        for k in range(self.n_comp):
            mlab.points3d(self.means[k, dim][0],
                          self.means[k, dim][1],
                          self.means[k, dim][2],
                          color=(1, 0., 0.),
                          scale_factor=0.05)
            plot_gaussian_mesh_on_tangent_plane(self.means[k, dim], self.covariances[k, :self.dim, :self.dim], color=(0.5, 0, 0.2))
        mlab.view(30, 120)
        mlab.show()

    def plot_gmm(self):
        # Pick 15 random datapoints from X to plot
        rand_idx = np.random.choice(np.arange(1, len(self.dataset.X)), size=15, replace=False, p=None)
        plot_data = self.dataset.X[rand_idx[0]].numpy()
        for i in rand_idx[1:]:
            plot_data = np.vstack([plot_data, self.dataset.X[i].numpy()])

        plot_means = np.empty((self.n_comp, 3))
        for i in range(plot_means.shape[0]):
            for j in range(plot_means.shape[1]):
                plot_means[i, j] = self.means[i, 0][j]

        temp = self.covariances[:, :self.dim, :self.dim]
        plot_covariances = np.empty((self.n_comp, 3))
        for i in range(plot_covariances.shape[0]):
            for j in range(plot_covariances.shape[1]):
                plot_covariances[i, j] = temp[i][j,j]

        return visualize_3d_gmm(points=plot_data, w=self.priors, 
                         mu=plot_means.T, stdev=plot_covariances.T, 
                         skill=self.dataset.skill, 
                         export_dir=self.skills_dir, 
                         export_type='gif')