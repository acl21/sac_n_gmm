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


class Agent(object):
    def __init__(
        self,
        calvin_env,
        state_type,
        target_tasks,
        task_sequential,
        sparse_rewards,
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
    ):
        self.cfg = OmegaConf.create(
            {
                "state_type": state_type,
                "target_tasks": target_tasks,
                "task_sequential": task_sequential,
                "sparse_rewards": sparse_rewards,
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

        self.cons_logger = logging.getLogger("Agent")
        if self.cfg.record:
            self.video_dir = os.path.join(self.cfg.exp_dir, "videos")
            os.makedirs(self.video_dir, exist_ok=True)

    def reset(self) -> None:
        """
        - Resets the environment
        - Moves the EE to a good start state
        - Updates the agent state through self.observation
        """
        self.observation = self.env.reset()
        self.observation = self.calibrate_EE_start_state(self.observation)
        self.episode_steps = 0

    def make_env(self):
        """
        Returns a task specific CALVIN env object
        """
        new_env_cfg = {**self.calvin_env}
        new_env_cfg["target_tasks"] = self.cfg.target_tasks
        new_env_cfg["sequential"] = self.cfg.task_sequential
        new_env_cfg["sparse_rewards"] = self.cfg.sparse_rewards
        env = TaskSpecificEnv(**new_env_cfg)
        env.state_type = self.cfg.state_type
        return env

    def get_state_dim(self):
        """
        Returns the state (what agent looks at) size
        """
        raise NotImplementedError

    def get_action_dim(self):
        """
        Return the size of action space as an int
        """
        return NotImplementedError

    def get_action_space(self):
        """
        Return a gym.spaces.Box() object of action space
        """
        raise NotImplementedError

    def get_action(self, actor, observation, strategy="stochastic"):
        """
        Interface to get action from SAC Actor,
        ready to be used in the environment
        Args:
            actor: actor model
            observation: what the actor model looks at to predict the next action
            strategy: action sampling strategy - "random", "zeros", "stochastic", "deterministic"
        """
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

    # def populate_parallel(
    #     self, steps: int = 1000, strategy: str = "random", nproc: int = 0
    # ):
    #     """
    #     Carries out several random steps through multiple environments
    #     run in parallel to fill up the replay buffer with experiences
    #     Args:
    #         steps: number of random steps to populate the buffer with
    #         nproc: number of proccesses to use to fill the replay buffer in parallel
    #     """
    #     # Create parallel environments
    #     nproc = os.cpu_count() if nproc == 0 else nproc
    #     envs = [self.make_env(seed) for seed in range(nproc)]
    #     envs = SubprocVecEnv(envs)
    #     # Perform random steps to populate the replay buffer
    #     step = 0
    #     observations = envs.reset()
    #     pbar = tqdm(total=steps)
    #     while step < steps:
    #         actions = self.get_stack_actions(observations, strategy)
    #         next_observations, rewards, dones, infos = envs.step(actions)
    #         next_observations = next_observations
    #         for i, done in enumerate(dones):
    #             # if 'TimeLimit.truncated' in infos[i]:
    #             #     done = done and not infos[i]['TimeLimit.truncated']
    #             next_obs = (
    #                 infos[i]["terminal_observation"] if done else next_observations[i]
    #             )
    #             replay_buffer.add_transition(
    #                 observations[i], actions[i], next_obs, rewards[i], dones[i]
    #             )
    #             observations[i] = next_observations[i]
    #             step += 1
    #             pbar.update(1)
    #     pbar.close()

    def play_step(self, actor, strategy="stochastic", replay_buffer=None):
        """
        Perform a step in the environment and add the transition
        tuple to the replay buffer
        Args:
            actor: actor model
            strategy: action sampling strategy - "random", "zeros", "stochastic", "deterministic"
            replay_buffer: replay buffer object
        """
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, actor):
        """
        Evaluate the actor in the environment
        """
        raise NotImplementedError

    def calibrate_EE_start_state(self, obs, error_margin=0.01, max_checks=15):
        """
        Samples a random but good starting point and moves the end effector to that point
        """
        desired_start = self.skill_ds[0].sample_start(
            size=1, sigma=0.05
        )  # From val_dataset
        count = 0
        state = np.append(desired_start, np.append(self.fixed_ori, -1))
        action = self.env.prepare_action(state, action_type="abs")
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
