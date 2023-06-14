from omegaconf import OmegaConf
import os
import gym
import torch
import numpy as np
from tqdm import tqdm
import logging

from examples.CALVINExperiment.envs.task_env import TaskSpecificEnv
from examples.CALVINExperiment.rl.networks.utils.misc import transform_to_tensor
from examples.CALVINExperiment.seqblend.CALVINSkill import CALVINSkill


class CALVINSeqblendToyRLAgent(object):
    def __init__(
        self,
        calvin_env,
        state_type,
        target_tasks,
        task_sequential,
        device,
        num_train_steps,
        num_seed_steps,
        eval_frequency,
        num_eval_episodes,
        accumulate_env_steps,
        max_episode_steps,
        record,
        render,
        wandb,
        exp_dir,
        skills_dir,
        demos_dir,
    ) -> None:
        self.cfg = OmegaConf.create(
            {
                "state_type": state_type,
                "target_tasks": target_tasks,
                "task_sequential": task_sequential,
                "num_train_steps": num_train_steps,
                "num_seed_steps": num_seed_steps,
                "eval_frequency": eval_frequency,
                "num_eval_episodes": num_eval_episodes,
                "accumulate_env_steps": accumulate_env_steps,
                "max_episode_steps": max_episode_steps,
                "record": record,
                "render": render,
                "wandb": wandb,
                "exp_dir": exp_dir,
            }
        )
        self.device = device

        # Environment
        self.calvin_env = calvin_env
        self.env = self.make_env()
        self.action_space = self.get_action_space()

        # Dynamical System
        self.skill_ds = [
            CALVINSkill(skill, i, demos_dir, skills_dir)
            for i, skill in enumerate(self.cfg.target_tasks)
        ]
        self.fixed_ori = self.skill_ds[0].dataset.fixed_ori
        self.dt = self.skill_ds[0].dataset.dt

        # Trackers
        # State variable
        self.observation = None
        # At any point in time, my agent can only perform self.cfg.max_episode_steps number of
        # "play_step"s in a given episode, this tracks that
        self.episode_steps = 0
        # This tracks total episodes done in an experiment
        self.episode_idx = 0
        # This tracks total "play_steps" taken in an experiment
        self.steps = 0
        # This tracks total environment steps taken in an experiment
        self.env_steps = 0
        # Agent resets - env and state variable self.observation
        self.reset()

        # Logging
        self.cons_logger = logging.getLogger("CALVINSeqblendRLAgent")
        if self.cfg.record:
            self.video_dir = os.path.join(self.cfg.exp_dir, "videos")
            os.makedirs(self.video_dir, exist_ok=True)

    def reset(self) -> None:
        """Resets the environment, moves the EE to a good start state and updates the agent state"""
        self.observation = self.env.reset()
        self.observation = self.calibrate_EE_start_state(self.observation)
        self.episode_steps = 0

    def make_env(self):
        new_env_cfg = {**self.calvin_env}
        new_env_cfg["target_tasks"] = self.cfg.target_tasks
        new_env_cfg["sequential"] = self.cfg.task_sequential
        env = TaskSpecificEnv(**new_env_cfg)
        env.state_type = self.cfg.state_type
        return env

    def get_state_dim(self):
        """Env obs size"""
        return self.observation.size

    def get_action_space(self):
        """For this task, just 1 (W ~ [0, 1]) is enough."""
        action_dim = 1
        return gym.spaces.Box(low=np.zeros(action_dim), high=np.ones(action_dim))

    def play_step(self, actor, strategy="stochastic", replay_buffer=None):
        """Perform a step in the environment and add the transition
        tuple to the replay buffer"""
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
            action = self.env.prepare_action(temp, type="abs")
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

    def populate_replay_buffer(self, actor, replay_buffer):
        """
        Carries out several steps through the environment to initially fill
        up the replay buffer with experiences from the GMM
        Args:
            steps: number of random steps to populate the buffer with
            strategy: strategy to follow to select actions to fill the replay buffer
        """
        self.cons_logger.info("Populating replay buffer with random warm up steps")
        for _ in tqdm(range(self.cfg.num_seed_steps)):
            self.play_step(actor=actor, strategy="random", replay_buffer=replay_buffer)

        replay_buffer.save()

    def get_action(self, actor, observation, strategy="stochastic"):
        """Interface to get action from SAC Actor,
        ready to be used in the environment"""
        if strategy == "random":
            return self.action_space.sample()
        elif strategy == "zeros":
            return np.zeros(self.action_space.shape)
        elif strategy == "stochastic":
            deterministic = False
        elif strategy == "deterministic":
            deterministic = True
        else:
            raise Exception("Strategy not implemented")
        observation = transform_to_tensor(observation, device=self.device)
        action, _ = actor.get_actions(
            observation, deterministic=deterministic, reparameterize=False
        )

        return action.detach().cpu().numpy()

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
                self.env.reset_recorded_frames()
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
                    action = self.env.prepare_action(temp, type="abs")
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
                video_path = self.env.save_recorded_frames(
                    outdir=self.video_dir,
                    fname=f"{self.episode_idx}_{self.steps}_{self.env_steps}",
                )
                self.env.reset_recorded_frames()
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

    def calibrate_EE_start_state(self, obs, error_margin=0.01, max_checks=15):
        """Samples a random starting point and moves the end effector to that point"""
        desired_start = self.skill_ds[0].sample_start(
            size=1, sigma=0.05
        )  # From val_dataset
        count = 0
        state = np.append(desired_start, np.append(self.fixed_ori, -1))
        action = self.env.prepare_action(state, type="abs")
        while np.linalg.norm(obs[:3] - desired_start) > error_margin:
            obs, _, _, _ = self.env.step(action)
            count += 1
            if count >= max_checks:
                # self.cons_logger.info(
                #     f"CALVIN is struggling to place the EE at the right initial pose. \
                #         Difference: {np.linalg.norm(obs - desired_start)}"
                # )
                break
        return obs
