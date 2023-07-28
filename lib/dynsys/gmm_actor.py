import copy
import logging
import numpy as np
from lib.dynsys.manifold_gmm import ManifoldGMM


class GMMSkillActor:
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
        self.skill_ds_original = None

    def train(self, retrain=False):
        if retrain:
            "Train all skills one by one"
            for idx, ds in enumerate(self.skill_ds):
                self.logger.info(f"Training Manifold GMM {idx}:{ds.skill}: Started")
                ds.train(self.logger)
                self.logger.info(f"Training Manifold GMM {idx}:{ds.skill}: Ended")
        else:
            self.logger.info("Loading previously trained Manifold GMM weights")
            self.load_params()
        self.skill_ds_original = copy.deepcopy(self.skill_ds)

    def load_params(self):
        for _, ds in enumerate(self.skill_ds):
            ds.load_params(self.logger)

    def act(self, x, skill_id):
        return self.skill_ds[skill_id].predict_dx(x)

    def update_params(self, delta, skill_id):
        self.skill_ds[skill_id] = copy.deepcopy(
            self.skill_ds_original[skill_id]
        ).update_params(delta)

    def reset_params(self, skill_id):
        self.skill_ds[skill_id] = copy.deepcopy(self.skill_ds_original[skill_id])

    def reset_all_params(self):
        self.skill_ds = copy.deepcopy(self.skill_ds_original)

    def sample_start(self, size=1, sigma=0.15):
        pass

    def sample_gaussian_norm_ball(self, reference_point, sigma, num_samples):
        pass
