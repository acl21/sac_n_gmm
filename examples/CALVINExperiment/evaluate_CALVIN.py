from distutils.command.config import config
import os
import sys
from pathlib import Path

import numpy as np
import hydra
from omegaconf import DictConfig

cwd_path = Path(__file__).absolute().parents[0]
parent_path = cwd_path.parents[0]

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, parent_path.as_posix()) 
sys.path.insert(0, os.path.join(cwd_path, 'calvin_env')) #CALVIN env 
sys.path.insert(0, cwd_path.parents[0].parents[0].as_posix()) # Root

import csv
import torch
from torch.utils.data import DataLoader

from envs.skill_env import SkillSpecificEnv
from SkillsSequencing.skills.mps.dynsys.CLFDS import CLFDS
from SkillsSequencing.skills.mps.dynsys.WSAQF import WSAQF
from SkillsSequencing.skills.mps.dynsys.FNN import SimpleNN
from SkillsSequencing.skills.mps.dynsys.CALVIN_DS import CALVINDynSysDataset

import logging
logger = logging.getLogger(__name__)

import pdb

class SkillEvaluator(object):
    """Python wrapper that allows you to evaluate learned DS skills 
    in the CALVIN environment.
    """

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        f = open(cfg.skills_list, "r")
        skill_set = f.read()
        self.skill_set = skill_set.split("\n")
        self.logger = logging.getLogger('SkillEvaluator')

    def evaluate(self, ds, dataset, max_steps=500, sampling_dt=1/30, render=False):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        succesful_rollouts, rollout_returns, rollout_lengths = 0, [], []
        # start_idx, end_idx = self.env.get_valid_columns()
        for _, (xi, d_xi) in enumerate(dataloader):
            self.logger.info(f'Test Trajectory {_+1}')
            x0 = xi.squeeze()[0, :].numpy()
            goal = xi.squeeze()[-1, :].numpy()
            rollout_return = 0
            observation = self.env.reset()
            current_state = observation[:3]
            action = np.array([x0[:3], observation[3:6], -1], dtype=object)
            # self.logger.info(f'Adjusting EE position to match the initial pose from the dataset')
            while np.linalg.norm(current_state - x0) > 0.005:
                observation, reward, done, info = self.env.step(action)
                current_state = observation[:3]
            x = current_state[:3]
            # self.logger.info(f'Simulating with DS')
            for step in range(max_steps):
                d_x = ds.reg_model.forward(torch.from_numpy(x-goal[:3]).float().unsqueeze(dim=0).unsqueeze(dim=0))
                d_x = d_x.detach().cpu().numpy().squeeze()
                d_x = sampling_dt * d_x
                action = np.append(d_x, np.append(np.zeros(3), -1))
                observation, reward, done, info = self.env.step(action)
                x = observation[:3]
                rollout_return += reward
                if render:
                    self.env.render()
                if done:
                    break
            if info["success"]:
                succesful_rollouts += 1
                self.logger.info('Success!')
            rollout_returns.append(rollout_return)
            rollout_lengths.append(step)
        acc = succesful_rollouts / len(dataset.X)
        return acc, np.mean(rollout_returns), np.mean(rollout_lengths)

    def run(self):
        skill_accs = {}
        for skill in self.skill_set:
            skill_model_dir = os.path.join(self.cfg.skills_dir, self.cfg.state_type, skill)
            clf_file = os.path.join(skill_model_dir, 'clf')
            reg_file = os.path.join(skill_model_dir, 'ds')

            # Get train and validation datasets
            val_dataset = CALVINDynSysDataset(skill=skill, state_type=self.cfg.state_type, \
                train=False, demos_dir=self.cfg.demos_dir, goal_centered=False)

            # Create and load models to evaluate
            dim = val_dataset.X.shape[-1]
            clf_model = WSAQF(dim=dim, n_qfcn=1)
            reg_model = SimpleNN(in_dim=dim, out_dim=dim, n_layers=(20, 20))
            clfds = CLFDS(clf_model, reg_model, rho_0=0.1, kappa_0=0.0001)
            clfds.load_models(clf_file=clf_file, reg_file=reg_file)

            logger.info(f'Evaluating {skill} skill on CALVIN environment')
            logger.info(f'Test/Val Data: {val_dataset.X.size()}')
            # Evaluate by simulating in the CALVIN environment
            acc, avg_return, avg_len = self.evaluate(clfds, val_dataset, max_steps=self.cfg.max_steps)
            skill_accs[skill] = [str(acc), str(avg_return), str(avg_len)]

        # Write accuracies to a file
        with open(os.path.join(cwd_path, 'logs/', f'skill_ds_acc_{self.cfg.state_type}.txt'), 'w') as f:
            writer = csv.writer(f)
            for row in skill_accs.items():
                writer.writerow(row)


@hydra.main(config_path="./config", config_name="eval_ds")
def main(cfg: DictConfig) -> None:
    new_env_cfg = {**cfg.calvin_env.env}
    new_env_cfg["tasks"] = cfg.calvin_env.tasks
    new_env_cfg.pop("_target_", None)
    new_env_cfg.pop("_recursive_", None)
    env = SkillSpecificEnv(**new_env_cfg)
    env.set_state_type(cfg.state_type)

    eval = SkillEvaluator(cfg, env)
    eval.run()

if __name__ == "__main__":
    main()