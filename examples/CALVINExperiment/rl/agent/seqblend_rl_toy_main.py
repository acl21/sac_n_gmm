import os
import gym
import torch
import numpy as np
from tqdm import tqdm
import logging

from examples.CALVINExperiment.rl.agent.base_agent import Agent


class CALVINSeqblendToyRLAgent(Agent):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Logging
        self.cons_logger = logging.getLogger("CALVINSeqblendToyRLAgent")

    def get_state_dim(self):
        """
        Specific to this toy agent
        """
        return self.env.get_obs_space().shape[0]

    def get_action_space(self):
        """
        For this toy task, just 1 (W ~ [0, 1]) is enough.
        """
        action_dim = 1
        return gym.spaces.Box(low=np.zeros(action_dim), high=np.ones(action_dim))

    def play_step(self, actor, strategy="stochastic", replay_buffer=None):
        """
        Perform a step in the environment and add the transition
        tuple to the replay buffer
        """
        weight = self.get_action(actor, self.observation, strategy)
        if weight <= 0.5:
            idx = 0
        else:
            idx = 1
        obs = self.observation
        total_reward = 0
        for _ in range(self.cfg.accumulate_env_steps):
            # Do nothing when you are at the goal
            dist_to_goal = np.linalg.norm(obs[:3] - self.skill_ds[idx].goal)
            if np.round(dist_to_goal, 2) <= 0.01:
                # Goal is reached, so dx should now be 0
                dx = np.zeros(self.skill_ds[idx].dim())
            else:
                dx = self.skill_ds[idx].predict_dx(obs[:3])
            new_x = obs[:3] + dx * self.dt
            temp = np.append(new_x, np.append(self.fixed_ori, -1))
            action = self.env.prepare_action(temp, action_type="abs")
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            self.env_steps += 1

            if done:
                break

        next_observation = obs
        replay_buffer.add(
            self.observation,
            weight,
            total_reward,
            next_observation,
            done,
        )
        self.observation = next_observation

        self.steps += 1
        self.episode_steps += 1
        if done or (self.episode_steps >= self.cfg.max_episode_steps):
            self.reset()
            done = True
            self.episode_idx += 1
        return total_reward, done, int(weight.item() > 0.5)

    @torch.no_grad()
    def evaluate(self, actor):
        """Evaluates the actor in the environment"""
        self.cons_logger.info("Evaluation episodes in process")
        succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
        saved_video_path = None
        gt = [0, 1] * self.cfg.num_eval_episodes
        pred = []
        # Choose a random episode to record
        rand_idx = np.random.randint(0, self.cfg.num_eval_episodes)
        for episode in tqdm(range(1, self.cfg.num_eval_episodes + 1)):
            episode_steps = 0
            episode_return = 0
            # Reset agent and self.observation
            self.reset()
            # Recording setup
            if self.cfg.record and (episode == rand_idx):
                self.env.reset_recording()
                self.env.record_frame(size=64)
            while episode_steps < self.cfg.max_episode_steps:
                weight = self.get_action(actor, self.observation, "deterministic")
                if weight <= 0.5:
                    idx = 0
                else:
                    idx = 1
                pred.append(idx)
                obs = self.observation
                for _ in range(self.cfg.accumulate_env_steps):
                    # Do nothing when you are at the goal
                    dist_to_goal = np.linalg.norm(obs[:3] - self.skill_ds[idx].goal)
                    if np.round(dist_to_goal, 2) <= 0.01:
                        # Goal is reached, so dx should now be 0
                        dx = np.zeros(self.skill_ds[idx].dim())
                    else:
                        dx = self.skill_ds[idx].predict_dx(obs[:3])
                    new_x = obs[:3] + dx * self.dt
                    temp = np.append(new_x, np.append(self.fixed_ori, -1))
                    action = self.env.prepare_action(temp, action_type="abs")
                    obs, reward, done, info = self.env.step(action)
                    episode_return += reward
                    if self.cfg.render:
                        self.env.render()
                    if self.cfg.record and (episode == rand_idx):
                        self.env.record_frame(size=64)
                    if done:
                        break
                self.observation = obs
                episode_steps += 1
                if done:
                    break
            if ("success" in info) and info["success"]:
                succesful_episodes += 1

            if self.cfg.record and (episode == rand_idx):
                video_path = self.env.save_recording(
                    outdir=self.video_dir,
                    fname=f"{self.episode_idx}_{self.steps}_{self.env_steps}",
                )
                self.env.reset_recording()
                if os.path.isfile(video_path):
                    saved_video_path = video_path
                else:
                    self.cons_logger.info("Env returned a path with an unsaved video!")
            episodes_returns.append(episode_return)
            episodes_lengths.append(episode_steps)
        # Reset agent before exiting evaluate()
        self.reset()
        # accuracy = succesful_episodes / self.cfg.num_eval_episodes
        accuracy = sum([pred[i] == gt[i] for i in range(len(gt))]) / len(gt)
        return (
            accuracy,
            np.mean(episodes_returns),
            np.mean(episodes_lengths),
            saved_video_path,
        )
