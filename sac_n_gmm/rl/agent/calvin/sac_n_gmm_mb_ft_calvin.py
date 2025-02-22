import logging
import hydra
from omegaconf import DictConfig
import os
from enum import Enum
import gym
import torch
import numpy as np
from tqdm import tqdm
import copy
from pytorch_lightning.utilities import rank_zero_only
from sac_n_gmm.rl.agent.base_agent import BaseAgent
from sac_n_gmm.rl.helpers.skill_actor import SkillActor
from sac_n_gmm.utils.utils import LinearDecay

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


OBS_KEY = "rgb_gripper"
# OBS_KEY = "robot_obs"

LETTERS_TO_SKILLS = {
    "A": "open_drawer",
    "B": "turn_on_lightbulb",
    "C": "move_slider_left",
    "D": "turn_on_led",
    "E": "close_drawer",
    "F": "turn_off_lightbulb",
    "G": "move_slider_right",
    "H": "turn_off_led",
}


class CALVIN_SACNGMM_MB_FT_Agent(BaseAgent):
    def __init__(
        self,
        calvin_env: DictConfig,
        datamodule: DictConfig,
        num_init_steps: int,
        num_eval_episodes: int,
        task: DictConfig,
        gmm: DictConfig,
        priors_change_range: float,
        mu_change_range: float,
        quat_change_range: float,
        adapt_cov: bool,
        mean_shift: bool,
        adapt_per_skill: int,
        exp_dir: str,
        root_dir: str,
        render: bool,
        record: bool,
        device: str,
        sparse_reward: bool,
        cem_cfg: DictConfig,
    ) -> None:
        super().__init__(
            env=calvin_env,
            num_init_steps=num_init_steps,
            num_eval_episodes=num_eval_episodes,
        )

        self.task = task
        task_order = [*task.order]
        self.task.skills = [LETTERS_TO_SKILLS[skill] for skill in task_order]
        self.task.max_steps = self.task.skill_max_steps * len(self.task.skills)

        # Environment
        self.env.set_task(self.task.skills)
        self.env.max_episode_steps = self.task.max_steps
        self.env.sparse_reward = sparse_reward

        # Refine parameters
        self.priors_change_range = priors_change_range
        self.mu_change_range = mu_change_range
        self.quat_change_range = quat_change_range
        self.adapt_cov = adapt_cov
        self.mean_shift = mean_shift
        self.adapt_per_skill = adapt_per_skill
        self.gmm_window = 16  # self.task.max_steps // (self.adapt_per_skill * len(self.task.skills))

        # One SkillActor per set of skills
        self.skill_actor = SkillActor(self.task)
        # The order of skills inside actor should always be the same as the order of skills in the SKILLS enum
        self.skill_actor.skill_names = self.task.skills
        # GMM
        self.skill_actor.make_skills(gmm)
        # Load GMM weights of each skill
        self.skill_actor.load_models()
        # Use Dataset to set skill parameters - goal, fixed_ori, pos_dt, ori_dt
        self.skill_actor.set_skill_params(datamodule.dataset)
        if "Manifold" in self.skill_actor.name:
            self.skill_actor.make_manifolds()
        self.initial_gmms = copy.deepcopy(self.skill_actor.skills)
        self.skill_params_stacked = torch.from_numpy(self.skill_actor.get_all_skill_params(self.initial_gmms))

        # Store skill info - starts, goals, fixed_ori, pos_dt, ori_dt
        # self.env.store_skill_info(self.skill_actor.skills)
        # # record setup
        self.video_dir = os.path.join(exp_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)
        self.render = render
        self.record = record

        # CEM
        self.cem_cfg = cem_cfg
        self._std_decay = LinearDecay(cem_cfg.max_std, cem_cfg.min_std, cem_cfg.std_step)
        self._horizon_decay = LinearDecay(1, 1, cem_cfg.horizon_step)

        self.reset()
        self.skill_id = 0

        self.root_dir = root_dir

        self.nan_counter = 0
        self.one_hot_skill_vector = False

    @torch.no_grad()
    def play_step(self, refine_actor, model, critic, strategy="stochastic", replay_buffer=None, device="cuda"):
        """Perform a step in the environment and add the transition
        tuple to the replay buffer"""
        # Change dynamical system
        self.skill_actor.copy_model(self.initial_gmms, self.skill_id)
        gmm_change = self.get_action(refine_actor, model, critic, self.skill_id, self.obs, strategy, device)
        self.update_gaussians(gmm_change, self.skill_id)
        previous_skill_id = self.skill_id
        # Act with the dynamical system in the environment
        gmm_reward = 0
        curr_obs = self.obs
        for _ in range(self.gmm_window):
            conn = self.env.isConnected()
            if not conn:
                done = False
                break
            env_action, is_nan = self.skill_actor.act(curr_obs["robot_obs"], self.skill_id)
            if is_nan:
                self.nan_counter += 1
                done = True
                log_rank_0("Nan in prediction, aborting episode")
            else:
                curr_obs, reward, done, info = self.env.step(env_action)
                self.episode_env_steps += 1
                self.total_env_steps += 1
                if reward > 0:
                    self.skill_id = (self.skill_id + 1) % len(self.task.skills)
                    if "success" in info and not info["success"]:
                        reward = 0
                gmm_reward += reward

            if done:
                break

        _, info = self.env._termination()
        if "success" in info and info["success"]:
            replay_buffer.add(self.obs, previous_skill_id, gmm_change, gmm_reward, curr_obs, previous_skill_id, done)
        else:
            replay_buffer.add(self.obs, previous_skill_id, gmm_change, gmm_reward, curr_obs, self.skill_id, done)
        self.obs = curr_obs

        self.episode_play_steps += 1
        self.total_play_steps += 1

        if done or not conn:
            self.reset()
            self.skill_id = 0
        return gmm_reward, done

    @torch.no_grad()
    def evaluate(self, refine_actor, model, critic, device="cuda"):
        """Evaluates the actor in the environment"""
        log_rank_0("Evaluation episodes in process")
        succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
        saved_video_path = None
        succesful_skill_ids = []
        # Choose a random episode to record
        rand_idx = np.random.randint(1, self.num_eval_episodes + 1)
        for episode in tqdm(range(1, self.num_eval_episodes + 1)):
            episode_return, episode_env_steps = 0, 0
            self.obs = self.env.reset()
            skill_id = 0
            # log_rank_0(f"Skill: {skill} - Obs: {self.obs['robot_obs']}")
            # Recording setup
            if self.record and (episode == rand_idx):
                self.env.reset_recording()
                self.env.record_frame(size=200)

            while episode_env_steps < self.task.max_steps:
                # Change dynamical system
                self.skill_actor.copy_model(self.initial_gmms, skill_id)
                gmm_change = self.get_action(refine_actor, model, critic, skill_id, self.obs, "deterministic", device)
                self.update_gaussians(gmm_change, skill_id)

                # Act with the dynamical system in the environment
                # x = self.obs["position"]

                for _ in range(self.gmm_window):
                    env_action, is_nan = self.skill_actor.act(self.obs["robot_obs"], skill_id)
                    if is_nan:
                        done = True
                        log_rank_0("Nan in prediction, aborting episode")
                    else:
                        self.obs, reward, done, info = self.env.step(env_action)
                        episode_return += reward
                        episode_env_steps += 1

                        if reward > 0:
                            succesful_skill_ids.append(skill_id)
                            skill_id = (skill_id + 1) % len(self.task.skills)
                    if self.record and (episode == rand_idx):
                        self.env.record_frame(size=200)
                    if self.render:
                        self.env.render()
                    if done:
                        break

                if done:
                    self.reset()
                    skill_id = 0
                    break

            if ("success" in info) and info["success"]:
                succesful_episodes += 1
            # Recording setup close
            if self.record and (episode == rand_idx):
                video_path = self.env.save_recording(
                    outdir=self.video_dir,
                    fname=f"PlaySteps{self.total_play_steps}_EnvSteps{self.total_env_steps }_{episode}",
                )
                self.env.reset_recording()
                saved_video_path = video_path

            episodes_returns.append(episode_return)
            episodes_lengths.append(episode_env_steps)
        accuracy = succesful_episodes / self.num_eval_episodes
        return (
            accuracy,
            np.mean(episodes_returns),
            np.mean(episodes_lengths),
            succesful_skill_ids,
            saved_video_path,
        )

    @torch.no_grad()
    def estimate_value(self, refine_actor, model, critic, state, ac, horizon):
        """Imagine a trajectory for `horizon` steps, and estimate the value."""
        value, discount = 0, 1
        for t in range(horizon):
            state, reward = model.imagine_step(state, ac[t])
            value += discount * reward
            discount *= self.cem_cfg.cem_discount
        action, _ = refine_actor.get_actions(state, deterministic=False, reparameterize=False)
        value += discount * torch.min(*[*critic(state, action)]).squeeze()
        return value

    @torch.no_grad()
    def plan(self, refine_actor, model, critic, actor_input, prev_mean=None, is_train=True, device="cuda"):
        cfg = self.cem_cfg
        step = self.total_env_steps
        horizon = int(self._horizon_decay(step))
        clamp_max = 0.999

        # Sample policy trajectories.
        z = actor_input.repeat(cfg.num_policy_traj, 1)
        policy_ac = []
        for t in range(horizon):
            actions, _ = refine_actor.get_actions(z)
            policy_ac.append(actions)
            z, _ = model.imagine_step(z, policy_ac[t])
        policy_ac = torch.stack(policy_ac, dim=0)

        # CEM optimization.
        z = actor_input.repeat(cfg.num_policy_traj + cfg.num_sample_traj, 1)
        mean = torch.zeros(horizon, policy_ac.shape[-1], device=device)
        std = 2 * torch.ones(horizon, policy_ac.shape[-1], device=device)
        if prev_mean is not None and horizon > 1 and prev_mean.shape[0] == horizon:
            mean[:-1] = prev_mean[1:]

        for _ in range(cfg.cem_iter):
            sample_ac = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                horizon, cfg.num_sample_traj, policy_ac.shape[-1], device=device
            )
            sample_ac = torch.clamp(sample_ac, -clamp_max, clamp_max)

            ac = torch.cat([sample_ac, policy_ac], dim=1)

            imagine_return = self.estimate_value(refine_actor, model, critic, z, ac, horizon).squeeze(-1)
            _, idxs = imagine_return.sort(dim=0)
            idxs = idxs[-cfg.num_elites :]
            elite_value = imagine_return[idxs]
            elite_action = ac[:, idxs]

            # Weighted aggregation of elite plans.
            score = torch.exp(cfg.cem_temperature * (elite_value - elite_value.max()))
            score = (score / score.sum()).view(1, -1, 1)
            new_mean = (score * elite_action).sum(dim=1)
            new_std = torch.sqrt(torch.sum(score * (elite_action - new_mean.unsqueeze(1)) ** 2, dim=1))

            mean = cfg.cem_momentum * mean + (1 - cfg.cem_momentum) * new_mean
            std = torch.clamp(new_std, self._std_decay(step), 2)

        # Sample action for MPC.
        score = score.squeeze().cpu().numpy()
        ac = elite_action[0, np.random.choice(np.arange(cfg.num_elites), p=score)]
        if is_train:
            ac += std[0] * torch.randn_like(std[0])
        return torch.clamp(ac, -clamp_max, clamp_max), mean

    def get_skill_vector(self, skill_id, device="cuda"):
        if type(skill_id) is int:  # When skill_id is of shape (Batch x 1)
            if self.one_hot_skill_vector:
                skill_vector = torch.eye(len(self.task.skills))[skill_id]
            else:
                skill_vector = self.skill_params_stacked[skill_id].squeeze(0)
        else:
            if self.one_hot_skill_vector:
                skill_vector = torch.eye(len(self.task.skills))[skill_id[:, 0].cpu().int()]
            else:
                skill_vector = self.skill_params_stacked[skill_id[:, 0].cpu().int()]
        return skill_vector.to(device)

    def get_state_from_observation(self, encoder, obs, skill_id, device="cuda"):
        skill_vector = self.get_skill_vector(skill_id, device=device)
        if isinstance(obs, dict):
            # Robot obs
            if "state" in obs:
                name = "state"
            elif "robot_obs" in obs:
                name = "robot_obs"
            # RGB obs
            if "rgb_gripper" in obs:
                x = obs["rgb_gripper"]
                if not torch.is_tensor(x):
                    x = torch.tensor(x).to(device)
                if len(x.shape) < 4:
                    x = x.unsqueeze(0)
                with torch.no_grad():
                    features = encoder(x)
                    # features = encoder({"obs": x.float()})
                # If position are part of the input state
                # if features is not None:
                #     fc_input = torch.cat((fc_input, features.squeeze()), dim=-1).to(device)
                # Else
                fc_input = features.squeeze()

            fc_input = torch.cat((fc_input, skill_vector.squeeze()), dim=-1).to(device)
            return fc_input.float()

        return skill_vector

    def get_action(self, actor, model, critic, skill_id, observation, strategy="stochastic", device="cuda"):
        """Interface to get action from SAC Actor,
        ready to be used in the environment"""
        actor.eval()
        model.eval()
        if strategy == "random":
            return self.get_action_space().sample()
        elif strategy == "zeros":
            return np.zeros(self.get_action_space().shape)
        elif strategy == "stochastic":
            deterministic = False
        elif strategy == "deterministic":
            deterministic = True
        elif strategy == "cem":
            critic.eval()
            skill_vector = self.get_skill_vector(skill_id, device)
            obs_tensor = torch.from_numpy(observation[OBS_KEY]).to(device)
            enc_ob = model.encoder({"obs": obs_tensor.float()}).squeeze(0)
            actor_input = torch.cat((enc_ob, skill_vector), dim=-1).to(device).float()
            action, _ = self.plan(actor, model, critic, actor_input, device=device)
            actor.train()
            model.train()
            critic.train()
            return action.detach().cpu().numpy()
        else:
            raise Exception("Strategy not implemented")
        with torch.no_grad():
            skill_vector = self.get_skill_vector(skill_id, device)
            obs_tensor = torch.from_numpy(observation[OBS_KEY]).to(device)
            enc_ob = model.encoder({"obs": obs_tensor.float()}).squeeze(0)
            actor_input = torch.cat((enc_ob, skill_vector), dim=-1).to(device).float()
            action, _ = actor.get_actions(actor_input, deterministic=deterministic, reparameterize=False)
        actor.train()
        model.train()
        return action.detach().cpu().numpy()
