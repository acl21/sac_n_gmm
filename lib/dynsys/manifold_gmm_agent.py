import os
import logging
import numpy as np
from pymanopt.manifolds import Euclidean, Product
from lib.dynsys.manifold_gmm import ManifoldGMM
from lib.dynsys.manifold_gmr import manifold_gmr
from lib.dynsys.calvin_dynsys_dataset import CALVINDynSysDataset


class ManifoldGMMSkillAgent:
    """
    This class handles all functions related to CALVIN skills
    """

    def __init__(self, cfg):
        self.skill_names = cfg.skills
        self.skills_dir = cfg.skills_dir
        self.skill_ds = [
            ManifoldGMM(skill=skill, cfg=cfg) for skill in self.skill_names
        ]
        self.logger = logging.getLogger(
            "SeqRefAgent.SeqRefSkillAgent.ManifoldGMMSkillAgent"
        )

    def train(self):
        "Train all skills one by one"
        for idx, ds in enumerate(self.skill_ds):
            self.logger.info(f"Training Manifold GMM {idx}:{ds.skill}: Started")
            ds.train(self.logger)
            self.logger.info(f"Training Manifold GMM {idx}:{ds.skill}: Ended")

    def load_params(self):
        for _, ds in enumerate(self.skill_ds):
            ds.load_params(self.logger)

    def predict_dx(self, x, skill_vector):
        idx = np.argmax(skill_vector)
        return self.skill_ds[idx].predict_dx(x)

    def sample_start(self, size=1, sigma=0.15):
        pass

    def sample_gaussian_norm_ball(self, reference_point, sigma, num_samples):
        pass
