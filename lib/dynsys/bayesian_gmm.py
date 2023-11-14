from sklearn.mixture import BayesianGaussianMixture
from gmr import GMM
from lib.dynsys.base_gmm import BaseGMM
import numpy as np
import logging
from pathlib import Path
import os
from lib.dynsys.calvin_dynsys_dataset import CALVINDynSysDataset


class BayesianGMM(BaseGMM):
    def __init__(self, skill, cfg):
        super(BayesianGMM, self).__init__(
            n_components=cfg.gmm_components,
            plot=cfg.plot_gmm,
            model_dir=os.path.join(cfg.skills_dir, skill, "bayesiangmm"),
        )

        self.name = "BayesianGMM"
        self.skill = skill
        self.random_state = np.random.RandomState(0)
        self.max_iter = 500
        self.bgmm = BayesianGaussianMixture(
            n_components=self.n_components, max_iter=self.max_iter
        )
        self.gmm = None

        self.dataset = CALVINDynSysDataset(
            skill=self.skill,
            train=cfg.dataset.train,
            demos_dir=cfg.dataset.demos_dir,
            goal_centered=cfg.dataset.goal_centered,
        )

        # Some params from the dataset
        self.start = self.dataset.start
        self.goal = self.dataset.goal
        self.fixed_ori = self.dataset.fixed_ori
        self.dt = self.dataset.dt

    def fit(self):
        self.set_data_params(self.dataset)
        self.bgmm = self.bgmm.fit(self.data)
        self.means, self.covariances, self.priors = (
            self.bgmm.means_,
            self.bgmm.covariances_,
            self.bgmm.weights_,
        )

        self.gmm = GMM(
            n_components=self.n_components,
            priors=self.priors,
            means=self.means,
            covariances=self.covariances,
            random_state=self.random_state,
        )

        # Save GMM params
        self.save_model()

        # Plot GMM
        if self.plot:
            outfile = self.plot_gmm(obj_type=False)

    def load_model(self):
        super().load_model()
        self.gmm = GMM(
            n_components=self.n_components,
            priors=self.priors,
            means=self.means,
            covariances=self.covariances,
            random_state=self.random_state,
        )

    def update_model(self, delta):
        super().update_model(delta)
        self.gmm.means = self.means
        self.gmm.priors = self.priors

    def copy_model(self, gmm):
        super().copy_model(gmm)
        self.gmm.means = self.means
        self.gmm.priors = self.priors

    def predict(self, x):
        cgmm = self.gmm.condition([0, 1, 2], x[:3].reshape(1, -1))
        dx = cgmm.sample_confidence_region(1, alpha=0.7).reshape(-1)
        return dx
