import os
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from examples.CALVINExperiment.rl.utils.replay_buffer import ReplayBuffer
from examples.CALVINExperiment.rl.utils.dataset import RLDataset

import hydra
import wandb


class SAC(pl.LightningModule):
    """Basic SAC implementation using PyTorch Lightning"""

    def __init__(
        self,
        input_dim,
        action_dim,
        device,
        discount,
        init_alpha,
        optimize_alpha,
        alpha_lr,
        alpha_betas,
        actor_lr,
        actor_betas,
        critic_lr,
        critic_betas,
        critic_tau,
        batch_size,
        replay_buffer_capacity,
        actor,
        critic,
        agent,
    ):
        super().__init__()
        device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.batch_size = batch_size
        self.optimize_alpha = optimize_alpha

        # Agent
        self.agent = hydra.utils.instantiate(agent)
        self.action_space = self.agent.get_action_space()
        state_dim = self.agent.get_state_dim()
        action_dim = self.action_space.shape[0]

        # Critic
        critic.input_dim = state_dim + action_dim
        self.critic = hydra.utils.instantiate(critic).to(device)
        self.critic_target = hydra.utils.instantiate(critic).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Actor
        actor.input_dim = state_dim
        actor.action_dim = self.action_space.shape[0]
        self.actor = hydra.utils.instantiate(actor).to(device)
        self.actor.action_space = self.action_space

        # Entropy
        # Set target entropy to -|A|
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(init_alpha)).to(device)
        self.log_alpha.requires_grad = True

        # Optimizers
        self.actor_lr, self.actor_betas = actor_lr, actor_betas
        self.critic_lr, self.critic_betas = critic_lr, critic_betas
        self.alpha_lr, self.alpha_betas = alpha_lr, alpha_betas
        (
            self.actor_optimizer,
            self.critic_optimizer,
            self.log_alpha_optimizer,
        ) = self.configure_optimizers()

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(
            max_capacity=int(replay_buffer_capacity),
        )
        self.replay_buffer.save_dir = os.path.join(
            self.agent.cfg.exp_dir, "replay_buffer/"
        )
        os.makedirs(self.replay_buffer.save_dir, exist_ok=True)

        # Populate Replay Buffer with Random Actions
        self.agent.populate_replay_buffer(self.actor, self.replay_buffer)

        # Logic values
        self.episode_idx = torch.zeros(1, requires_grad=False)
        self.episode_return = 0
        self.episode_length = 0

        # Action tracker (within an episode)
        # self.action_tracker = [-1 for x in range(self.agent.cfg.max_episode_steps)]
        # self.episode_actions = self.action_tracker
        # self.track_actions = True
        # self.action_tracker_n = 0

        # To keep track of evaluation calls
        self.last_eval_episode_idx = 0

        # PyTorch Lightning
        self.automatic_optimization = False

    def track_running_avg(self, tracker, tracker_n, new_values):
        """
        Running average
        """
        for i, x in enumerate(tracker):
            tracker[i] = ((tracker_n) * tracker[i] + new_values[i]) / (tracker_n + 1)
        return tracker

    def training_step(self, batch, batch_idx):
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch received
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        """
        reward, self.episode_done, action = self.agent.play_step(
            self.actor, "stochastic", self.replay_buffer
        )
        # self.episode_actions[self.episode_length] = action
        self.episode_return += reward
        self.episode_length += 1

        batch = self.overwrite_batch(batch)
        optimizers = self.optimizers()
        self.optimize_networks(batch, optimizers)
        self.soft_update(self.critic_target, self.critic, self.critic_tau)

        if self.episode_done:
            # When an episode ends, log episode metrics
            wandb_obj = self.logger.experiment
            wandb_logs = {
                "train/episode_return": self.episode_return,
                "train/episode_length": self.episode_length,
                "train/episode_number": self.episode_idx.item(),
                "train/env_steps": self.agent.env_steps,
            }
            wandb_obj.log(wandb_logs)
            self.episode_return, self.episode_length = 0, 0
            self.episode_idx += 1

            # If gap to last evaluation is more than "eval_frequency" episodes
            if (
                self.episode_idx.item() - self.last_eval_episode_idx
            ) >= self.agent.cfg.eval_frequency:
                self.eval_step()
                self.last_eval_episode_idx = self.episode_idx.item()

            # Log average actions across episodes
            # if self.track_actions:
            #     self.action_tracker = self.track_running_avg(
            #         self.action_tracker, self.action_tracker_n, self.episode_actions
            #     )
            #     self.action_tracker_n += 1
            #     self.episode_actions = [
            #         -1 for x in range(self.agent.cfg.max_episode_steps)
            #     ]
            #     data = [
            #         [x + 1, y]
            #         for (x, y) in zip(
            #             range(len(self.action_tracker)), self.action_tracker
            #         )
            #     ]
            #     table = wandb.Table(data=data, columns=["Episode_Step", "Weight"])
            #     wandb_obj.log(
            #         {
            #             "train/weight": wandb.plot.line(
            #                 table,
            #                 "Episode_Step",
            #                 "Weight",
            #                 title="Episode_Step vs Weight",
            #             )
            #         }
            #     )
            #     self.action_tracker = [
            #         -1 for x in range(self.agent.cfg.max_episode_steps)
            #     ]

    def eval_step(self):
        """
        This function is called every time a training epoch ends.
        One training epoch = replay_buffer size / batch_size iterations
        It evaluates actor (when appropriate) by simulating
        in the environment.
        """
        eval_return = float("-inf")
        eval_accuracy = float("-inf")
        # Log every episode end
        wandb_obj = self.logger.experiment
        wandb_logs = {}
        (
            eval_accuracy,
            eval_return,
            eval_length,
            eval_video_path,
        ) = self.agent.evaluate(self.actor)
        wandb_logs["eval/accuracy"] = eval_accuracy
        wandb_logs["eval/avg_episode_return"] = eval_return
        wandb_logs["eval/avg_episode_length"] = eval_length
        wandb_logs["eval/env_steps"] = self.agent.env_steps
        wandb_logs["eval/episode_number"] = self.episode_idx.item()

        # Log the video GIF to wandb if exists
        if eval_video_path is not None:
            wandb_logs["eval/video"] = wandb.Video(
                eval_video_path, fps=15, format="gif"
            )

        # Save the replay buffer when you evaluate
        self.replay_buffer.save()

        wandb_obj.log(wandb_logs)

        # Monitored metric to save model
        self.log("eval_episode_return", eval_return, on_epoch=True)
        self.log("eval_accuracy", eval_accuracy, on_epoch=True)
        self.log("env_steps", self.agent.env_steps, on_epoch=True)

    def configure_optimizers(self):
        """Initialize optimizers"""

        actor_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.actor.parameters()),
            lr=self.actor_lr,
            betas=self.actor_betas,
        )

        critic_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.critic.parameters()),
            lr=self.critic_lr,
            betas=self.critic_betas,
        )

        log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.alpha_lr, betas=self.alpha_betas
        )
        return (actor_optimizer, critic_optimizer, log_alpha_optimizer)

    def optimize_networks(self, batch, optimizers):
        actor_optimizer, critic_optimizer, alpha_optimizer = optimizers[:3]
        actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(
            batch, alpha_optimizer
        )
        actor_optimizer.zero_grad()
        self.manual_backward(actor_loss)
        actor_optimizer.step()

        critic_loss = self.compute_critic_loss(batch)
        critic_optimizer.zero_grad()
        self.manual_backward(critic_loss)
        critic_optimizer.step()

        self.log("losses/critic_loss", critic_loss, on_step=True)
        self.log("losses/actor_loss", actor_loss, on_step=True)
        self.log("losses/alpha_loss", alpha_loss, on_step=True)
        self.log("losses/env_steps", self.agent.env_steps, on_step=True)

    def compute_actor_and_alpha_loss(self, batch, alpha_optimizer):
        batch_observations = batch[0]
        policy_actions, curr_log_pi = self.actor.get_actions(
            batch_observations,
            deterministic=False,
            reparameterize=True,
        )
        alpha_loss = -(
            self.log_alpha * (curr_log_pi + self.target_entropy).detach()
        ).mean()

        if self.optimize_alpha:
            alpha_optimizer.zero_grad()
            self.manual_backward(alpha_loss)
            alpha_optimizer.step()

        alpha = self.log_alpha.exp()
        self.log("losses/alpha", alpha, on_step=True)

        q1, q2 = self.critic(batch_observations, policy_actions)
        Q_value = torch.min(q1, q2)
        actor_loss = (alpha * curr_log_pi - Q_value).mean()
        return actor_loss, alpha_loss

    def compute_critic_loss(self, batch):
        (
            batch_observations,
            batch_actions,
            batch_rewards,
            batch_next_observations,
            batch_dones,
        ) = batch

        with torch.no_grad():
            next_actions, next_log_pi = self.actor.get_actions(
                batch_next_observations,
                deterministic=False,
                reparameterize=False,
            )
            q1_next_target, q2_next_target = self.critic_target(
                batch_next_observations, next_actions
            )
            q_next_target = torch.min(q1_next_target, q2_next_target)
            alpha = self.log_alpha.exp()
            q_target = batch_rewards + (1 - batch_dones) * self.discount * (
                q_next_target - alpha * next_log_pi
            )

        # Bellman loss
        q1_pred, q2_pred = self.critic(batch_observations, batch_actions.float())
        bellman_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
        return bellman_loss

    @staticmethod
    def soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def train_dataloader(self):
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.replay_buffer, self.batch_size)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True
        )
        return dataloader

    def overwrite_batch(self, batch):
        """Verifies if everything is as expected inside a batch"""
        obs, actions, rew, next_obs, dones = batch
        # Verifying batch shape
        if len(rew.shape) == 1:
            rew = rew.unsqueeze(-1)
        if len(dones.shape) == 1:
            dones = dones.unsqueeze(-1)

        # Verifying input type
        rew = rew.float()
        actions = actions.float()
        dones = dones.int()
        if not isinstance(obs, dict):
            obs = obs.float()
            next_obs = next_obs.float()

        # Verifying device
        if rew.device != self.device:
            rew = rew.to(self.device)
        if actions.device != self.device:
            actions = actions.to(self.device)
        if dones.device != self.device:
            dones = dones.to(self.device)
        batch = obs, actions, rew, next_obs, dones
        return batch
