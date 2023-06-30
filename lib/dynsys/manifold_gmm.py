import os
import numpy as np
from pymanopt.manifolds import Euclidean, Sphere, Product

from lib.dynsys.manifold_clustering import (
    manifold_k_means,
    manifold_gmm_em,
)
from lib.dynsys.manifold_gmr import manifold_gmr
from lib.dynsys.utils.plot_utils import visualize_3d_gmm
from lib.dynsys.calvin_dynsys_dataset import CALVINDynSysDataset


class ManifoldGMM(object):
    def __init__(self, skill, cfg):
        self.skill = skill
        self.n_comp = cfg.gmm_components
        self.plot = cfg.plot_gmm
        self.wandb = cfg.wandb

        # Data and Manifold
        self.dataset = CALVINDynSysDataset(
            skill=self.skill,
            train=cfg.dataset.train,
            state_type=cfg.dataset.input_type,
            demos_dir=cfg.dataset.demos_dir,
            goal_centered=cfg.dataset.goal_centered,
            is_quaternion=cfg.dataset.is_quaternion,
            ignore_bad_demos=cfg.dataset.ignore_bad_demos,
        )
        self.state_type = cfg.dataset.input_type
        self.dim = self.dataset.X.shape[-1]
        self.manifold = self.create_manifold()

        # Goal state
        self.goal = self.dataset.goal

        # GMM
        self.means = None
        self.covariances = None
        self.priors = None
        self.assignments = None

        # Directory where GMM params are saved
        self.skills_dir = os.path.join(
            cfg.skills_dir, self.state_type, self.skill, "manifoldgmm"
        )
        os.makedirs(self.skills_dir, exist_ok=True)

    def create_manifold(
        self,
    ):
        if self.state_type in ["pos", "joint"]:
            in_manifold = Euclidean(self.dim)
            out_manifold = Euclidean(self.dim)
        elif self.state_type == "ori":
            in_manifold = Sphere(self.dim)
            out_manifold = Sphere(self.dim)
        elif self.state_type == "pos_ori":
            manifold = None
        manifold = Product([in_manifold, out_manifold])
        return manifold

    def prepare_training_data(self):
        # Stack position and velocity data
        demos_xdx = [
            np.hstack([self.dataset.X[i], self.dataset.dX[i]])
            for i in range(self.dataset.X.shape[0])
        ]
        # Stack demos
        demos = demos_xdx[0]
        for i in range(1, self.dataset.X.shape[0]):
            demos = np.vstack([demos, demos_xdx[i]])

        X = demos[:, : self.dim]
        Y = demos[:, self.dim :]

        data = np.empty((X.shape[0], 2), dtype=object)
        for n in range(X.shape[0]):
            data[n] = [X[n], Y[n]]
        return data

    def load_params(self, logger, filename="/weights.npz"):
        weights_file = self.skills_dir + filename
        if not os.path.exists(weights_file):
            raise FileNotFoundError(
                f"Could not find Manifold GMM params at {weights_file}"
            )
        else:
            f"Loading Manifold GMM params for {self.skill} from {weights_file}"
        gmm = np.load(weights_file)
        gmm.allow_pickle = True
        self.means = np.array(gmm["means"])
        self.covariances = np.array(gmm["covariances"])
        self.priors = np.array(gmm["priors"])

    def save_params(self, logger, filename="/weights.npz"):
        weights_file = self.skills_dir + filename
        np.savez(
            weights_file,
            means=self.means,
            covariances=self.covariances,
            priors=self.priors,
        )
        logger.info(f"Saved Manifold GMM params of {self.skill} at {weights_file}")

    def train(self, logger):
        # Data
        data = self.prepare_training_data()

        # K-Means
        logger.info("Manifold K-Means: Started")
        km_means, km_assignments = manifold_k_means(
            self.manifold, np.copy(data), nb_clusters=self.n_comp, logger=logger
        )
        logger.info("Manifold K-Means: Ended")
        # GMM
        logger.info("Manifold GMM with K-Means priors: Started")
        init_covariances = np.concatenate(
            self.n_comp * [np.eye(self.dim + self.dim)[None]], 0
        )
        init_priors = np.zeros(self.n_comp)
        for k in range(self.n_comp):
            init_priors[k] = np.sum(km_assignments == k) / len(km_assignments)
        self.means, self.covariances, self.priors, self.assignments = manifold_gmm_em(
            self.manifold,
            np.copy(data),
            self.n_comp,
            initial_means=km_means,
            initial_covariances=init_covariances,
            initial_priors=init_priors,
            logger=logger,
        )
        logger.info("Manifold GMM with K-Means priors: Ended")

        # Reshape means from (n_components, 2) to (n_components, 2, state_size)
        self.means = self.get_reshaped_means()

        # Save GMM params
        self.save_params(logger)

        # Plot GMM
        if self.plot:
            outfile = self.plot_gmm()

        if self.wandb:
            import wandb

            config = {"n_comp": self.n_comp, "state_type": self.state_type}
            wandb.init(
                project="seqref",
                entity="in-ac",
                name=f"pretrain.gmm.{self.dataset.skill}",
                config=config,
            )
            wandb.log({"GMM-Viz": wandb.Video(outfile)})
            wandb.finish()

        # Release dataset object (memory efficient?)
        self.dataset = None

    def predict_dx(self, x):
        """
        Goal centers the input x and returns dx
        """
        dx, _, __ = manifold_gmr(
            (x - self.goal).reshape(1, -1),
            self.manifold,
            self.means,
            self.covariances,
            self.priors,
        )
        return dx

    def get_reshaped_means(self):
        """Reshape means from (n_comp, 2) to (n_comp, 2, state_size)"""
        new_means = np.empty((self.n_comp, 2, self.dim))
        for i in range(new_means.shape[0]):
            for j in range(new_means.shape[1]):
                new_means[i, j, :] = self.means[i][j]
        return new_means

    def plot_gmm_mlab(self, input_space=True):
        from mayavi import mlab
        from lib.dynsys.utils.plot_sphere_mayavi import (
            plot_sphere,
            plot_gaussian_mesh_on_tangent_plane,
        )

        if input_space:
            dim = 0
        else:
            dim = 1
        nb_data = self.dataset.X[0].shape[0]
        X = np.concatenate(self.data[:, dim]).reshape(
            self.data.shape[0], len(self.data[0, 0])
        )
        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(700, 700))
        fig = mlab.gcf()
        mlab.clf()
        plot_sphere(figure=fig)
        # Plot data on the sphere
        for p in range():
            mlab.points3d(
                X[p * nb_data : (p + 1) * nb_data, 0],
                X[p * nb_data : (p + 1) * nb_data, 1],
                X[p * nb_data : (p + 1) * nb_data, 2],
                color=(0.0, 0.0, 0.0),
                scale_factor=0.03,
            )
        # Plot Gaussians
        for k in range(self.n_comp):
            mlab.points3d(
                self.means[k, dim][0],
                self.means[k, dim][1],
                self.means[k, dim][2],
                color=(1, 0.0, 0.0),
                scale_factor=0.05,
            )
            plot_gaussian_mesh_on_tangent_plane(
                self.means[k, dim],
                self.covariances[k, : self.dim, : self.dim],
                color=(0.5, 0, 0.2),
            )
        mlab.view(30, 120)
        mlab.show()

    def plot_gmm(self):
        means = self.means

        # Pick 15 random datapoints from X to plot
        points = self.dataset.X[:, :, :3]
        rand_idx = np.random.choice(np.arange(0, len(points)), size=15)
        points = np.vstack(points[rand_idx, :, :])
        means = np.vstack(self.means[:, 0])
        covariances = self.covariances[:, :3, :3]

        return visualize_3d_gmm(
            points=points,
            priors=self.priors,
            means=means,
            covariances=covariances,
            save_dir=self.skills_dir,
        )
