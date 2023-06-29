import os
import logging
import numpy as np
from pymanopt.manifolds import Euclidean, Product
from lib.dynsys.manifold_gmm import ManifoldGMM
from lib.dynsys.manifold_gmr import manifold_gmr
from lib.dynsys.calvin_dynsys_dataset import CALVINDynSysDataset


class ManifoldGMMAgent:
    """
    This class handles all functions related to CALVIN skills
    """

    def __init__(self, cfg):
        self.skill_names = cfg.skills
        self.skills_dir = cfg.skills_dir
        self.skill_ds = [
            ManifoldGMM(skill=skill, cfg=cfg) for skill in self.skill_names
        ]

    def load_skill_ds_params(self):
        skill_file = os.path.join(
            self.skills_dir, "pos", self.skill_name, "gmm", "gmm_params.npz"
        )

        if not os.path.exists(skill_file):
            raise FileNotFoundError(f"Skill GMM Params not found at {skill_file}")
        else:
            self.logger.info(f"Loading GMM params from {skill_file}")
        gmm = np.load(skill_file)
        gmm.allow_pickle = True
        self.means = np.array(gmm["gmm_means"])
        self.covariances = np.array(gmm["gmm_covariances"])
        self.priors = np.array(gmm["gmm_priors"])

    def predict_dx(self, x):
        dx, _, __ = manifold_gmr(
            (x - self.goal).reshape(1, -1),
            self.manifold,
            self.means,
            self.covariances,
            self.priors,
        )
        return dx

    def dim(self):
        return self.dim_

    def sample_start(self, size=1, sigma=0.15):
        rand_idx = np.random.choice(self.val_starts_good_ones, size=1)[0]
        return self.val_dataset.X.numpy()[rand_idx, 3, :3]
        # start = self.dataset.start
        # sampled = self.sample_gaussian_norm_ball(start, sigma, size)
        # if size == 1:
        #     return sampled[0]
        # else:
        #     return sampled

    def sample_gaussian_norm_ball(self, reference_point, sigma, num_samples):
        samples = []
        for _ in range(num_samples):
            # Step 1: Sample from standard Gaussian distribution
            offset = np.random.randn(3)

            # Step 2: Normalize the offset
            normalized_offset = offset / np.linalg.norm(offset)

            # Step 3: Scale the normalized offset
            scaled_offset = normalized_offset * np.random.normal(0, sigma)

            # Step 4: Translate the offset
            sampled_point = reference_point + scaled_offset

            samples.append(sampled_point)

        return samples
