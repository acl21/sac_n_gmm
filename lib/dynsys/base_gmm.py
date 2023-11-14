import logging
import os
import sys
from pathlib import Path
from lib.dynsys.utils.plot_utils import visualize_3d_gmm
import pybullet

cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[0]
root = sac_gmm_path.parents[0]

# This is to access the locally installed repo clone when using slurm
sys.path.insert(0, sac_gmm_path.as_posix())  # sac_gmm
sys.path.insert(0, root.as_posix())  # Root

import numpy as np


logger = logging.getLogger(__name__)


def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class BaseGMM(object):
    """Gaussian Mixture Model.

    Type 1: GMM with position as input and velocity as output
    Type 2: GMM with position as input and next position as output
    Type 3: GMM with position as input and next position and next orientation as output
    Type 4: One GMM with position as input and velocity as output and
            another GMM with position as input and orientation as output
    Type 5: One GMM with position as input and next position as output and
            another GMM with position as input and orientation as output

    Parameters
    ----------
    n_components : int
        Number of components that compose the GMM.

    priors : array-like, shape (n_components,), optional
        Weights of the components.

    means : array-like, shape (n_components, n_features), optional
        Means of the components.

    covariances : array-like, shape (n_components, n_features, n_features), optional
        Covariances of the components.

    """

    def __init__(
        self,
        n_components=3,
        priors=None,
        means=None,
        covariances=None,
        plot=None,
        model_dir=None,
    ):
        # GMM
        self.n_components = n_components
        self.priors = priors
        self.means = means
        self.covariances = covariances
        self.plot = plot
        self.model_dir = model_dir
        self.pos_dt = 0.02
        self.ori_dt = 0.05
        self.goal = None
        self.start = None
        self.fixed_ori = None

        self.name = "GMM"

        # Data
        self.dataset = None
        self.data = None

        if self.priors is not None:
            self.priors = np.asarray(self.priors)
        if self.means is not None:
            self.means = np.asarray(self.means)
        if self.covariances is not None:
            self.covariances = np.asarray(self.covariances)

        os.makedirs(self.model_dir, exist_ok=True)

    def fit(self, dataset):
        """
        fits a GMM on demonstrations
        Args:
            dataset: skill demonstrations
        """
        raise NotImplementedError

    def predict(self, x):
        """
        Predict function for GMM type 1
        """
        raise NotImplementedError

    def set_data_params(self, dataset):
        self.dataset = dataset
        self.data = self.preprocess_data()

    def preprocess_data(self):
        data = None
        # Data size
        data_size = self.dataset.X_pos.shape[0]

        # Horizontal stack demos
        # Stack X_pos and dX_pos data
        demos_xdx = [
            np.hstack([self.dataset.X_pos[i], self.dataset.dX_pos[i]])
            for i in range(self.dataset.X_pos.shape[0])
        ]

        # Vertical stack demos
        demos = demos_xdx[0]
        for i in range(1, data_size):
            demos = np.vstack([demos, demos_xdx[i]])

        X_pos = demos[:, :3]
        dX_pos = demos[:, 3:6]

        data = np.hstack((X_pos, dX_pos))

        # Data for the second GMM
        return data

    def save_model(self):
        filename = self.model_dir + "/weights.npy"
        np.save(
            filename,
            {
                "priors": self.priors,
                "mu": self.means,
                "sigma": self.covariances,
            },
        )
        log_rank_0(f"Saved GMM params at {filename}")

    def load_model(self):
        filename = self.model_dir + "/weights.npy"
        log_rank_0(f"Loading GMM params from {filename}")

        gmm = np.load(filename, allow_pickle=True).item()

        self.priors = np.array(gmm["priors"])
        self.means = np.array(gmm["mu"])
        self.covariances = np.array(gmm["sigma"])

    def copy_model(self, gmm_obj):
        """Copies GMM params to self from the input GMM class object

        Args:
            gmm (BaseGMM|ManifoldGMM|BayesianGMM): GMM class object

        Returns:
            None
        """
        self.priors = np.copy(gmm_obj.priors)
        self.means = np.copy(gmm_obj.means)
        self.covariances = np.copy(gmm_obj.covariances)

    def model_params(self, cov=False):
        """Returns GMM priors and means as a flattened vector

        Args:
            None

        Returns:
            params (np.array): GMM params flattened
        """
        priors = self.priors
        means = self.means.flatten()
        params = np.concatenate((priors, means), axis=-1)
        return params

    def update_model(self, delta):
        """Updates GMM parameters by given delta changes (i.e. SAC's output)

        Args:
            delta (dict): Changes given by the SAC agent to be made to the GMM parameters

        Returns:
            None
        """
        # Priors
        if "priors" in delta:
            delta_priors = delta["priors"].reshape(self.priors.shape)
            self.priors += delta_priors
            self.priors[self.priors < 0] = 0
            self.priors /= self.priors.sum()

        # Means
        if "mu" in delta:
            delta_means = delta["mu"].reshape(self.means.shape)
            self.means += delta_means

    def get_params_size(self):
        return self.priors.size, self.means.size, self.covariances.size

    def plot_gmm(self, obj_type=True):
        if "Bayesian" in self.name:
            self.reshape_params(to="gmr-specific")
        points = self.dataset.X_pos.numpy()
        rand_idx = np.random.choice(np.arange(0, len(points)), size=15)
        points = np.vstack(points[rand_idx, :, :])
        means = np.vstack(self.means[:, 0])
        covariances = self.covariances[:, :3, :3]

        return visualize_3d_gmm(
            points=points,
            priors=self.priors,
            means=means,
            covariances=covariances,
            save_dir=self.model_dir,
        )

    def reshape_params(self, to="generic"):
        """Reshapes model params to/from generic/gmr-specific shapes.
        E.g., For N GMM components, S state size, generic shapes are
        self.priors = (N,);
        self.means = (N, 2*S);
        self.covariances = (N, 2*S, 2*S)

        Gmr-specific: self.means = (N, 2, S)
        """
        # priors and covariances already match shape
        shape = None
        if to == "generic":
            shape = (self.n_components, 2 * 3)
        else:
            shape = (self.n_components, 2, 3)
        self.means = self.means.reshape(shape)

    def get_reshaped_means(self):
        """Reshape means from (n_components, 2) to (n_components, 2, state_size)"""
        new_means = np.empty((self.n_components, 2, 3))
        for i in range(new_means.shape[0]):
            for j in range(new_means.shape[1]):
                new_means[i, j, :] = self.means[i][j]
        return new_means

    def get_reshaped_data(self):
        reshaped_data = None
        reshaped_data = np.empty((self.data.shape[0], 2), dtype=object)
        for n in range(self.data.shape[0]):
            reshaped_data[n] = [self.data[n, :3], self.data[n, 3:]]
            return reshaped_data

    def predict_dx(self, x):
        dx_pos = self.predict(x[:3] - self.goal)
        return np.append(dx_pos, np.append(np.zeros(3), -1))
