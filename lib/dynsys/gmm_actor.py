import copy
import logging
import numpy as np
from lib.dynsys.bayesian_gmm import BayesianGMM
from lib.dynsys.utils.batch_gmm import BatchGMM
import torch


class GMMSkillActor:
    """
    This class handles all functions related to CALVIN skills
    """

    def __init__(self, cfg):
        self.skill_names = cfg.skills
        self.skills_dir = cfg.skills_dir
        self.skill_ds = [
            BayesianGMM(skill=skill, cfg=cfg) for skill in self.skill_names
        ]
        self.logger = logging.getLogger(
            "SeqRefAgent.SeqRefSkillAgent.BayesianGMMSkillAgent"
        )
        self.skill_ds_original = None

        self.skill_priors = torch.from_numpy(
            np.array([skill.priors for skill in self.skill_ds])
        )
        self.skill_means = torch.from_numpy(
            np.array([skill.means for skill in self.skill_ds])
        )
        self.skill_covariances = torch.from_numpy(
            np.array([skill.covariances for skill in self.skill_ds])
        )

    def train(self, retrain=False):
        if retrain:
            "Train all skills one by one"
            for idx, ds in enumerate(self.skill_ds):
                self.logger.info(f"Fitting BayesianGMM {idx}:{ds.skill}: Started")
                ds.fit()
                self.logger.info(f"Fitting BayesianGMM {idx}:{ds.skill}: Ended")
        else:
            self.logger.info("Loading previously trained BayesianGMM weights")
            self.load_model()
        self.skill_ds_original = copy.deepcopy(self.skill_ds)

    def load_model(self):
        for _, ds in enumerate(self.skill_ds):
            ds.load_model()
        self.skill_ds_original = copy.deepcopy(self.skill_ds)

    def act(self, x, skill_id):
        return self.skill_ds[skill_id].predict_dx(x)

    def update_model(self, delta, skill_id):
        # Always reset GMM weights to original before refining
        self.reset_model(skill_id)
        self.skill_ds[skill_id].update_model(delta)

    def reset_model(self, skill_id):
        self.skill_ds[skill_id] = copy.deepcopy(self.skill_ds_original[skill_id])

    def reset_all_models(self):
        self.skill_ds = copy.deepcopy(self.skill_ds_original)

    @torch.no_grad()
    def batch_actions(self, batch_x, batch_rv, batch_skill_ids, skill_horizon):
        """
        Batch acts skill_horizon times
        """
        batch_size = batch_x.shape[0]
        out = torch.zeros((batch_size, skill_horizon, batch_x.shape[1]))

        batch_priors = self.skill_priors[batch_skill_ids]
        batch_means = self.skill_means[batch_skill_ids]
        batch_covariances = self.skill_covariances[batch_skill_ids]
        # Batch refine (only means)
        batch_means += batch_rv.reshape(batch_means.shape)

        # Batch predict
        for i in range(skill_horizon):
            batch_dx = batch_predict(
                self.skill_ds[0].n_components,
                batch_priors,
                batch_means,
                batch_covariances,
                batch_x,
                self.skill_ds[0].random_state,
            )
            out[:, i, :] = batch_dx
            batch_x += batch_dx * self.skill_ds[0].dt

        return out


def batch_predict(
    n_components, batch_priors, batch_means, batch_covariances, batch_x, random_state
):
    """
    Batch Predict function for BayesianGMM

    Along the batch dimension, you have different means, covariances, and priors and input.
    The function outputs the predicted delta x for each batch.
    """
    batch_priors = None
    batch_means = None
    batch_covariances = None
    batch_condition = BatchGMM(
        n_components=n_components,
        priors=batch_priors,
        means=batch_means,
        covariances=batch_covariances,
        random_state=random_state,
    ).condition([0, 1, 2], batch_x)
    return batch_condition.one_sample_confidence_region(alpha=0.7)
