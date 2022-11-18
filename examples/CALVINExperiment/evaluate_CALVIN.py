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

import wandb
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

    def evaluate(self, ds, dataset, max_steps=500, sampling_dt=2/30, render=False, record=False):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        succesful_rollouts, rollout_returns, rollout_lengths = 0, [], []
        start_idx, end_idx = self.env.get_valid_columns()
        for idx, (xi, d_xi) in enumerate(dataloader):
            if (idx % 5 == 0) or (idx == len(dataset)):
                self.logger.info(f'Test Trajectory {idx+1}/{len(dataset)}')
            x0 = xi.squeeze()[0, :].numpy()
            goal = xi.squeeze()[-1, :].numpy()
            rollout_return = 0
            observation = self.env.reset()
            current_state = observation[start_idx:end_idx]
            action = self.env.prepare_action(x0, type='abs')
            # self.logger.info(f'Adjusting EE position to match the initial pose from the dataset')
            while np.linalg.norm(current_state - x0) > 0.005:
                observation, reward, done, info = self.env.step(action)
                current_state = observation[start_idx:end_idx]
            x = current_state
            # pdb.set_trace()
            # self.logger.info(f'Simulating with DS')
            if record:
                self.logger.info(f'Recording Robot Camera Obs')
                self.env.record()
            for step in range(max_steps):
                delta_x = ds.reg_model.forward(torch.from_numpy(x-goal).float().unsqueeze(dim=0).unsqueeze(dim=0))
                delta_x = delta_x.detach().cpu().numpy().squeeze()
                d_x = sampling_dt * delta_x
                # pdb.set_trace()
                action = self.env.prepare_action(d_x, type='rel')
                observation, reward, done, info = self.env.step(action)
                x = observation[start_idx:end_idx]
                rollout_return += reward
                if record:
                    self.env.record()
                if render:
                    self.env.render()
                if done:
                    break
            if record:
                self.logger.info(f'Saving Robot Camera Obs')
                video_path = self.env.save_and_reset_recording()
                if self.cfg.wandb:
                    if info["success"]:
                        status = 'Success'
                    else:
                        status = 'Fail'
                    wandb.log({f"{self.env.skill_name} {status} {self.env.count}":wandb.Video(video_path, fps=30, format="gif")})
            if info["success"]:
                succesful_rollouts += 1
                self.logger.info('Success!')
            rollout_returns.append(rollout_return)
            rollout_lengths.append(step)
        acc = succesful_rollouts / len(dataset.X)
        if self.cfg.wandb:
            wandb.config.update({'val dataset size': len(dataset.X)})
            wandb.log({'skill':self.env.skill_name, 'accuracy': acc*100, \
                       'average_return': np.mean(rollout_returns), \
                       'average_traj_len': np.mean(rollout_lengths)})
        return acc, np.mean(rollout_returns), np.mean(rollout_lengths)

    def run(self):
        skill_accs = {}
        for skill in self.skill_set:
            if self.cfg.wandb:
                config = {'state_type': self.cfg.state_type, \
                        'sampling_dt': self.cfg.sampling_dt, \
                        'max steps': self.cfg.max_steps}
                wandb.init(project="ds-evaluation", entity="in-ac", config=config, name=f'{skill}_{self.cfg.state_type}')
            self.env.set_skill(skill)
            skill_model_dir = os.path.join(self.cfg.skills_dir, self.cfg.state_type, skill)
            clf_file = os.path.join(skill_model_dir, 'clf')
            reg_file = os.path.join(skill_model_dir, 'ds')

            # Get train and validation datasets
            val_dataset = CALVINDynSysDataset(skill=skill, state_type=self.cfg.state_type, \
                                              train=False, demos_dir=self.cfg.demos_dir, goal_centered=False)
                                            #   dt=self.cfg.sampling_dt, sampling_dt=self.cfg.sampling_dt)

            # Create and load models to evaluate
            dim = val_dataset.X.shape[-1]
            clf_model = WSAQF(dim=dim, n_qfcn=1)
            reg_model = SimpleNN(in_dim=dim, out_dim=dim, n_layers=(20, 20))
            clfds = CLFDS(clf_model, reg_model, rho_0=0.1, kappa_0=0.0001)
            clfds.load_models(clf_file=clf_file, reg_file=reg_file)

            logger.info(f'Evaluating {skill} skill with {self.cfg.state_type} input on CALVIN environment')
            logger.info(f'Test/Val Data: {val_dataset.X.size()}')
            # Evaluate by simulating in the CALVIN environment
            acc, avg_return, avg_len = self.evaluate(clfds, val_dataset, max_steps=self.cfg.max_steps, \
                                                     render=self.cfg.render, record=self.cfg.record, \
                                                     sampling_dt=self.cfg.sampling_dt)
            skill_accs[skill] = [str(acc), str(avg_return), str(avg_len)]
            self.env.count = 0
            if self.cfg.wandb:
                wandb.finish()

        # Write accuracies to a file
        with open(os.path.join(self.env.outdir, f'skill_ds_acc_{self.cfg.state_type}.txt'), 'w') as f:
            writer = csv.writer(f)
            for row in skill_accs.items():
                writer.writerow(row)


@hydra.main(config_path="./config", config_name="eval_ds")
def main(cfg: DictConfig) -> None:
    new_env_cfg = {**cfg.calvin_env.env}
    new_env_cfg["use_egl"] = False
    new_env_cfg["show_gui"] = False
    new_env_cfg["use_vr"] = False
    new_env_cfg["use_scene_info"] = True
    new_env_cfg["tasks"] = cfg.calvin_env.tasks
    new_env_cfg.pop("_target_", None)
    new_env_cfg.pop("_recursive_", None)

    env = SkillSpecificEnv(**new_env_cfg)
    env.set_state_type(cfg.state_type)
    env.set_outdir(hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir'])

    eval = SkillEvaluator(cfg, env)
    eval.run()

if __name__ == "__main__":
    main()