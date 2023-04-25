import os
import sys
import time
import hydra
import torch
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from scipy.special import softmax
from examples.CALVINExperiment.envs.task_env import TaskSpecificEnv
from examples.CALVINExperiment.seqblend.CALVINSkill import CALVINSkill
from examples.CALVINExperiment.rl.logger import Logger
from examples.CALVINExperiment.rl.replay_buffer import ReplayBuffer
import examples.CALVINExperiment.rl.utils as rl_utils
from SkillsSequencing.utils.utils import prepare_torch

cwd_path = Path(__file__).absolute().parents[0]
calvin_exp_path = cwd_path.parents[0]
root = calvin_exp_path.parents[0]
sys.path.insert(0, calvin_exp_path.as_posix()) # CALVINExperiment
sys.path.insert(0, root.as_posix()) # Root

device = prepare_torch()


class CALVINSeqBlendRL(object):
    """
    This class is used to sequence and blend the CALVIN skills.
    """
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.skill_names = self.cfg.target_tasks
        self.skill_ds = [CALVINSkill(skill, i, cfg.demos_dir, cfg.skills_dir) for i, skill in enumerate(self.skill_names)]
        self.logger = Logger(self.env.outdir,
                             save_tb=self.cfg.log_save_tb,
                             log_frequency=self.cfg.log_frequency,
                             agent='sac')

        self.device = torch.device(self.cfg.device)
        self.cfg.agent.obs_dim = 3 # position
        self.cfg.agent.action_dim = self.env.action_space.shape[0]
        self.cfg.agent.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(self.cfg.agent)
        self.replay_buffer = ReplayBuffer(self.cfg.agent.obs_dim,
                                          self.cfg.agent.action_dim,
                                          int(self.cfg.replay_buffer_capacity),
                                          self.device)
        self.acc_steps = self.cfg.accumulate_steps
        self.fixed_ori = np.array(self.cfg.fixed_ori)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0

            if self.cfg.record:
                self.env.reset_recorded_frames()
                self.env.record_frame()

            while not done:
                with rl_utils.eval_mode(self.agent):
                    weights = self.agent.act(obs[:3], sample=False)
                    weights = softmax(weights)
                for _ in range(self.acc_steps):
                    dx = np.hstack([skill.predict_dx(obs[:3]) for skill in self.skill_ds])
                    # dx = weights * dx
                    dx = weights[0]*dx[:3] + weights[1]*dx[3:]
                    new_x = obs[:3] + dx * self.cfg.dt
                    temp = np.append(new_x, np.append(self.fixed_ori, -1))
                    action = self.env.prepare_action(temp, type='abs')
                    obs, reward, done, _ = self.env.step(action)
                    episode_reward += reward

                    if self.cfg.record:
                        self.env.record_frame()
                    if self.cfg.render:
                        self.env.render()
                    if done:
                        break

            if self.cfg.record:
                video_path = self.env.save_recorded_frames()
                self.env.reset_recorded_frames()

            average_episode_reward += episode_reward
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                weights = self.env.action_space.sample()
            else:
                with rl_utils.eval_mode(self.agent):
                    weights = self.agent.act(obs, sample=True)
            weights = softmax(weights)
            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            old_obs = obs[:3]
            for _ in range(self.acc_steps):
                dx = np.hstack([skill.predict_dx(obs[:3]) for skill in self.skill_ds])
                # dx = weights * dx
                dx = weights[0]*dx[:3] + weights[1]*dx[3:]
                new_x = obs[:3] + dx*self.cfg.dt
                temp = np.append(new_x, np.append(self.fixed_ori, -1))
                action = self.env.prepare_action(temp, type='abs')
                obs, reward, done, info = self.env.step(action)
                next_obs = obs[:3]

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(old_obs, weights, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(version_base='1.1', config_path='../config', config_name='train_sac')
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

    seqblend = CALVINSeqBlendRL(cfg, env)
    seqblend.run()

if __name__ == "__main__":
    main()