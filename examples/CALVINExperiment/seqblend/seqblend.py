import os
import sys
import hydra
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from examples.CALVINExperiment.envs.task_env import TaskSpecificEnv
from examples.CALVINExperiment.seqblend.CALVINExp import CALVINExp
from examples.CALVINExperiment.seqblend.CALVINSkill import CALVINSkillComplex
from SkillsRefining.qpnet.spec_datasets import SkillDataset
from SkillsRefining.qpnet import qpnet_policies as policy_classes
from SkillsRefining.utils.utils import prepare_torch

cwd_path = Path(__file__).absolute().parents[0]
calvin_exp_path = cwd_path.parents[0]
root = calvin_exp_path.parents[0]
sys.path.insert(0, calvin_exp_path.as_posix()) # CALVINExperiment
sys.path.insert(0, root.as_posix()) # Root

device = prepare_torch()


class CALVINSeqBlend(object):
    """
    This class is used to sequence and blend the CALVIN skills.
    """
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.skill_names = self.cfg.target_tasks
        self.exp = CALVINExp(self.skill_names, self.cfg.demos_dir, self.cfg.skills_dir)
        self.skill_dim = sum([self.exp.skill_list[i].dataset.X.numpy().shape[-1] for i in range(len(self.exp.skill_list))])
        self.logger = logging.getLogger('CALVINSeqBlend')

        if self.cfg.loss_weights is None:
            self.loss_weights = np.ones(len(self.cfg.target_tasks))
        else:
            self.loss_weights = np.array(self.cfg.loss_weights)

        # Updating policy when given else use it for testing
        if self.cfg.policy_path is not None:
            if os.path.exists(self.cfg.policy_path) and not self.cfg.update_policy and not self.cfg.eval_policy:
                self.logger.info('Remove old policy!')
                os.remove(self.cfg.policy_path)
        else:
            self.cfg.policy_path = os.path.join(self.cfg.policy_dir, 'policy.pth')

    def run(self):
        """
        Train and evaluate sequencing-blending policy.
        """
        Xt_d, dXt_d, desired_ = self.exp.get_taskspace_training_data()
        batch_skill = CALVINSkillComplex(self.skill_dim, self.skill_dim, self.exp.skill_list, skill_cluster_idx=None)

        timestamps = np.linspace(0, 1, self.cfg.timesteps - 1)
        feat = timestamps[:, np.newaxis]
        dataset = SkillDataset(feat, Xt_d, dXt_d, desired_)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

        # Get the policy type and parameters
        policy_config = {'dim': self.skill_dim,
                        'fdim': 1,
                        'skill': batch_skill}
        policy_type = getattr(policy_classes, self.cfg.policy_name)
        constraints = {'eqn': None, 'ineqn': None}

        # Create the policy
        policy = policy_type(**policy_config, **constraints)
        policy.to(device)
        if not os.path.exists(self.cfg.policy_path):
            # Train the policy
            # If the policy has a full weight matrix, we first train a policy with diagonal weight matrix
            if policy_type == policy_classes.SkillFullyWeightedPolicy:
                diag_policy = policy_classes.SkillDiagonalWeightedPolicy(**policy_config, **constraints)
                diag_policy.to(device)
                diag_policy = policy_classes.train_policy(diag_policy, dataloader,
                                                        loss_weights=self.loss_weights,
                                                        model_path=self.cfg.policy_path,
                                                        learning_rate=self.cfg.lr,
                                                        max_epochs=self.cfg.max_epochs,
                                                        consider_spec_train=False)
                # The diagonal part of the full weight matrix is then initialized with the pretrained diagonal policy
                # The full weight matrix is then trained starting at this initial point
                policy.QDiagNet = diag_policy.Qnet

            policy = policy_classes.train_policy(policy, dataloader,
                                                loss_weights=self.loss_weights,
                                                model_path=self.cfg.policy_path,
                                                learning_rate=self.cfg.lr,
                                                max_epochs=self.cfg.max_epochs,
                                                consider_spec_train=False)
        else:
            # If the policy already exists, load it
            policy.load_policy(self.cfg.policy_path)
            policy.skill = batch_skill
            policy.to(device)

        if self.cfg.eval_policy:
            # Test the policy
            self.exp.test_policy(policy, self.cfg, self.env)

@hydra.main(version_base='1.1', config_path='../config', config_name='seqblend')
def main(cfg: DictConfig) -> None:
    hydra_out_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']

    new_env_cfg = {**cfg.calvin_env.env}
    new_env_cfg["use_egl"] = False
    new_env_cfg["show_gui"] = False
    new_env_cfg["use_vr"] = False
    new_env_cfg["use_scene_info"] = True
    new_env_cfg["tasks"] = cfg.calvin_env.tasks
    new_env_cfg.pop("_target_", None)
    new_env_cfg.pop("_recursive_", None)
    new_env_cfg['target_tasks'] = cfg.target_tasks
    new_env_cfg['sequential'] = cfg.task_sequential

    env = TaskSpecificEnv(**new_env_cfg)
    env.state_type = cfg.state_type
    env.set_outdir(hydra_out_dir)

    cfg.policy_dir = hydra_out_dir
    seqblend = CALVINSeqBlend(cfg, env)
    seqblend.run()

if __name__ == "__main__":
    main()