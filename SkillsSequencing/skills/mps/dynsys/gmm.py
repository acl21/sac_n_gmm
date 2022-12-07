import numpy as np
from pymanopt.manifolds import Euclidean, Sphere, Product

from SkillsSequencing.skills.mps.gmr.manifold_clustering import manifold_k_means, manifold_gmm_em
from SkillsSequencing.skills.mps.gmr.manifold_gmr import manifold_gmr

import wandb
import logging

class ManifoldGMM(object):
    def __init__(self, n_components=3, plot_with_mlab=False):
        self.n_comp = n_components
        self.plot_with_mlab = plot_with_mlab
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
        demos_np = demos_xdx[0]
        for i in range(1, dataset.X.shape[0]):
            demos_np = np.vstack([demos_np, demos_xdx[i]])

        X = demos_np[:, :self.dim]
        Y = demos_np[:, self.dim:]

        if normalize:
            if self.state_type == 'pos':
                # Normalize to have range [-1, 1]
                X = 2*(X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0))-1
                Y = 2*(Y-np.min(Y,axis=0))/(np.max(Y, axis=0)-np.min(Y,axis=0))-1
            elif self.state_type == 'ori':
                # Normalize to have range [-1, 1]
                X = 2*(X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0))-1
                Y = 2*(Y-np.min(Y,axis=0))/(np.max(Y, axis=0)-np.min(Y,axis=0))-1
                # Unit Norm
                X = X / np.linalg.norm(X, axis=1)[:, None]
                Y = Y / np.linalg.norm(Y, axis=1)[:, None]

        data = np.empty((X.shape[0], 2), dtype=object)
        for n in range(X.shape[0]):
            data[n] = [X[n], Y[n]]
        return data

    def load_params(self, filename='/gmm_params.npz'):
        gmm = np.load(self.skills_dir + filename)
        gmm.allow_pickle = True
        self.means = np.array(gmm['gmm_means'])
        self.covariances = np.array(gmm['gmm_covariances'])
        self.priors = np.array(gmm['gmm_priors'])

    def save_params(self, filename='/gmm_params.npz'):
        np.savez(self.skills_dir + filename, gmm_means=self.means, \
                 gmm_covariances=self.covariances, \
                 gmm_priors=self.priors)

    def set_data_params(self, dataset):
        self.dataset = dataset
        self.state_type = self.dataset.state_type
        self.dim = self.dataset.X.numpy().shape[-1]
        self.manifold = self.make_manifold(self.dim)
        self.data = self.preprocess_data(dataset, normalize=False)

    def train(self, dataset, wandb_flag=False):
        # Dataset
        self.set_data_params(dataset)

        # K-Means
        km_means, km_assignments = manifold_k_means(self.manifold, self.data, \
                                                    nb_clusters=self.n_comp)
        # GMM
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
        if self.plot_with_mlab:
            self.plot_gmm_mlab(input_space=True)
            self.plot_gmm_mlab(input_space=False)
        else:
            self.plot_gmm_matplotlib(input_space=True)
            self.plot_gmm_matplotlib(input_space=False)

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

    def plot_gmm_matplotlib(self, input_space=True):
        pass
