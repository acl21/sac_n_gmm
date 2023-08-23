import pickle
import gzip
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import gym.spaces
import imageio
import matplotlib.pyplot as plt
import wandb

from rolf.algorithms import BaseAgent
from rolf.algorithms.dataset import ReplayBufferEpisode, SeqSampler
from rolf.utils import Logger, Info, StopWatch, LinearDecay
from rolf.utils.pytorch import optimizer_cuda, count_parameters
from rolf.utils.pytorch import copy_network, soft_copy_network
from rolf.utils.pytorch import to_tensor, RandomShiftsAug, AdamAMP
from rolf.networks.distributions import TanhNormal, mc_kl
from rolf.networks.dreamer import DenseDecoderTanh, ActionDecoder, Decoder
from rolf.networks.tdmpc_model import TDMPCModel, Encoder, LSTMEncoder

from lib.networks.tpmpc_model import CustomTDMPCModel
from lib.dynsys.gmm_actor import GMMSkillActor
from lib.dynsys.utils.refine import (
    get_meta_ac_space,
    get_refine_dict,
)
from seqref_rollout import SeqRefRolloutRunner

import multiprocessing as mp
import time


class SeqRefMetaSkillAgent(BaseAgent):
    """High-level + Low-level agent for SeqRef."""

    def __init__(self, cfg, ob_space, meta_ac_space, skill_ac_space):
        super().__init__(cfg, ob_space)
        self._ob_space = ob_space
        self._meta_ac_space = meta_ac_space
        self._skill_ac_space = skill_ac_space
        self._use_amp = cfg.precision == 16
        self._dtype = torch.float16 if self._use_amp else torch.float32
        self._std_decay = LinearDecay(cfg.max_std, cfg.min_std, cfg.std_step)
        self._horizon_decay = LinearDecay(1, cfg.n_skill, cfg.horizon_step)
        self._update_iter = 0
        self._meta_ac_dim = self._meta_ac_space.shape[0]
        self._skill_ac_dim = 3
        self._ob_dim = gym.spaces.flatdim(ob_space)
        self._refine = False

        torch.set_default_dtype(self._dtype)
        self.model = CustomTDMPCModel(
            cfg,
            self._ob_space,
            self._meta_ac_dim,
            self._skill_ac_dim * cfg.skill_dim,
            self._dtype,
        )
        self.model_target = CustomTDMPCModel(
            cfg,
            self._ob_space,
            self._meta_ac_dim,
            self._skill_ac_dim * cfg.skill_dim,
            self._dtype,
        )
        copy_network(self.model_target, self.model)
        self.meta_actor = ActionDecoder(
            cfg.state_dim,
            self._meta_ac_dim,
            [cfg.num_units] * cfg.num_layers,
            cfg.dense_act,
            cfg.log_std,
        )
        self.skill_actor = GMMSkillActor(cfg.pretrain)
        self.decoder = Decoder(cfg.decoder, cfg.state_dim, self._ob_space)
        self.to(self._device)

    def state_dict(self):
        return {
            "actor": self.meta_actor.state_dict(),
            "model": self.model.state_dict(),
            "model_target": self.model_target.state_dict(),
            "decoder": self.decoder.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self.model.load_state_dict(ckpt["model"])
        self.model_target.load_state_dict(ckpt["model_target"])
        self.decoder.load_state_dict(ckpt["decoder"])
        if "actor" in ckpt.keys():
            self.meta_actor.load_state_dict(ckpt["actor"])
        self.to(self._device)

    @property
    def ac_space(self):
        return (self._seq_space, self._ref_space)

    @torch.no_grad()
    def estimate_value(self, obs, state, ac, dyn_ac, horizon, skill_id, skill_horizon):
        """Imagine a trajectory for `horizon` steps, and estimate the value."""
        value, discount = 0, 1
        for t in range(horizon):
            state, reward = self.model.imagine_step(state, ac[t], dyn_ac[t])
            value += discount * reward
            discount *= self._cfg.rl_discount
        next_rv = self.meta_actor.act(state)
        value += discount * torch.min(*self.model.critic(state, next_rv))
        return value

    @torch.no_grad()
    def plan(self, ob, skill_id, prev_mean=None, is_train=True):
        """Plan given an observation `ob`."""
        cfg = self._cfg
        horizon = int(self._horizon_decay(self._step))

        state = self.model.encoder(ob)
        # Sample policy trajectories
        obs = ob["ob"].repeat(cfg.num_policy_traj, 1)
        z = state.repeat(cfg.num_policy_traj, 1)
        policy_ac = []
        policy_rv = []
        for t in range(horizon):
            rv = self.meta_actor.act(z)
            ac = self.get_refined_skill_actions(
                obs.cpu().numpy(),
                rv.cpu().numpy(),
                torch.tensor(skill_id).repeat((obs.shape[0], 1)).cpu().numpy(),
                cfg.skill_horizon,
            )
            policy_ac.append(ac)
            policy_rv.append(rv)
            z, _ = self.model.imagine_step(z, None, policy_ac[t])
            obs = self.decoder(z)  # TODO: Check this with Iman
        policy_ac = torch.stack(policy_ac, dim=0)
        policy_rv = torch.stack(policy_rv, dim=0)

        # CEM optimization
        z = state.repeat(cfg.num_policy_traj + cfg.num_sample_traj, 1)
        obs = ob["ob"].repeat(cfg.num_policy_traj + cfg.num_sample_traj, 1)
        mean = torch.zeros(horizon, self._meta_ac_dim, device=self._device)
        std = 2.0 * torch.ones(horizon, self._meta_ac_dim, device=self._device)
        if prev_mean is not None and horizon > 1 and prev_mean.shape[0] == horizon:
            mean[:-1] = prev_mean[1:]

        for _ in range(cfg.cem_iter):
            sample_rv = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                horizon, cfg.num_sample_traj, self._meta_ac_dim, device=self._device
            )
            sample_rv = torch.clamp(sample_rv, -0.999, 0.999)
            rv = torch.cat([sample_rv, policy_rv], dim=1)
            # start = time.time()
            sample_ac = self.get_refined_skill_actions(
                obs[: cfg.num_sample_traj].cpu().numpy(),
                sample_rv.squeeze(0).cpu().numpy(),
                torch.tensor(skill_id).repeat((cfg.num_sample_traj, 1)).cpu().numpy(),
                cfg.skill_horizon,
            ).unsqueeze(0)
            # end = time.time()
            # print("Method Runtime: ", end - start)

            # sample_ac = torch.clamp(sample_ac, -0.999, 0.999) # TODO: Is this necessary?
            ac = torch.cat([sample_ac, policy_ac], dim=1)

            imagine_return = self.estimate_value(
                obs, z, rv, ac, horizon, skill_id, cfg.skill_horizon
            ).squeeze(-1)
            _, idxs = imagine_return.sort(dim=0)
            idxs = idxs[-cfg.num_elites :]
            elite_value = imagine_return[idxs]
            elite_action = rv[:, idxs]

            # Weighted aggregation of elite plans.
            score = torch.exp(cfg.cem_temperature * (elite_value - elite_value.max()))
            score = (score / score.sum()).view(1, -1, 1)
            new_mean = (score * elite_action).sum(dim=1)
            new_std = torch.sqrt(
                torch.sum(score * (elite_action - new_mean.unsqueeze(1)) ** 2, dim=1)
            )

            mean = cfg.cem_momentum * mean + (1 - cfg.cem_momentum) * new_mean
            std = torch.clamp(new_std, self._std_decay(self._step), 2)

        # Sample action for MPC.
        score = score.squeeze().cpu().numpy()
        rv = elite_action[0, np.random.choice(np.arange(cfg.num_elites), p=score)]
        if is_train:
            rv += std[0] * torch.randn_like(std[0])
        return torch.clamp(rv, -0.999, 0.999), mean

    @torch.no_grad()
    def meta_act(self, ob, skill_id, mean=None, is_train=True, warmup=False):
        """
        Returns action and the actor's activation given an observation `ob`.
        #TODO: Make meta_act work with Batch input :(
        """
        ob = ob.copy()
        for k, v in ob.items():
            ob[k] = np.expand_dims(v, axis=0).copy()

        self.model.eval()
        self.meta_actor.eval()
        with torch.autocast(self._cfg.device, enabled=self._use_amp):
            ob = to_tensor(ob, self._device, self._dtype)
            ob = self.preprocess(ob)
            # act purely based on the policy
            if self._cfg.phase == "pretrain" or warmup or not self._cfg.use_cem:
                feat = self.model.encoder(ob)
                ac = self.meta_actor.act(feat, deterministic=not is_train)
                ac = ac.cpu().numpy().squeeze(0)
            # act based on CEM planning
            else:
                # TODO: Do we have to make plan() batch supported? Since they
                # always use it in rollouts
                # Logger.info("Planning starts")
                ac, mean = self.plan(ob, skill_id, mean, is_train)
                ac = ac.cpu().numpy()
                # Logger.info("Planning ends")
            ac = gym.spaces.unflatten(self._meta_ac_space, ac)
        self.model.train()
        self.meta_actor.train()
        return ac, mean

    def skill_act(self, obs, refine_vector, skill_id, is_train=False):
        """Takes x, performs horizon steps with skill_id-th skill, returns stacked dx-es"""
        if refine_vector is not None:
            self.refine(refine_vector.cpu().numpy(), skill_id)
        dx_pos, dx_ori = self.skill_actor.act(obs.copy(), skill_id)
        ac = np.append(
            dx_pos, np.append(dx_ori, -1)
        )  # pos, ori, gripper_width i.e., size=7
        ac = gym.spaces.unflatten(self._skill_ac_space, ac)
        return ac

    def refine(self, refine_vector, skill_id):
        refine_dict = get_refine_dict(self._cfg, refine_vector)
        self.skill_actor.refine_params(refine_dict, skill_id)

    @torch.no_grad()
    def get_refined_skill_actions_parallel(
        self, ob, refine_vector, skill_id, skill_horizon
    ):
        """
        Input
            ob: numpy array (B, 21)
            refine_vector: numpy array (B, self._meta_ac_dim)
            skill_id: numpy array (B, 1)
            skill_horizon: int
        Returns tensor (B, 30) stacked actions by performing the
        skill_id-th skill (refined by given refine_vector) for skill_horizon steps
        """
        splits = 8  # min([os.cpu_count(), ob.shape[0]])
        cfg_list = [copy.copy(self._cfg) for _ in range(splits)]
        skill_actor_list = [copy.deepcopy(self.skill_actor) for _ in range(splits)]
        ob_list = list(np.array_split(ob, splits))
        rv_list = list(np.array_split(refine_vector, splits))
        skill_id_list = list(np.array_split(skill_id, splits))
        skill_horizon_list = [skill_horizon for _ in range(splits)]
        skill_ac_dim_list = [self._skill_ac_dim for _ in range(splits)]
        arguments = list(
            zip(
                cfg_list,
                skill_actor_list,
                ob_list,
                rv_list,
                skill_id_list,
                skill_horizon_list,
                skill_ac_dim_list,
            )
        )

        pool = mp.Pool(processes=splits)
        start = time.time()
        stacked_acs = pool.starmap(refined_skill_actions, arguments)
        end = time.time()
        print("Starmap Runtime: ", end - start)
        pool.close()
        pool.join()  # TODO: Check if this slows the main thread down

        return torch.from_numpy(np.concatenate(stacked_acs)).to(
            dtype=self._dtype, device=self._device
        )

    @torch.no_grad()
    def get_refined_skill_actions(self, ob, refine_vector, skill_id, skill_horizon):
        """
        Input
            ob: tensor (B, 21)
            refine_vector: tensor (B, self._meta_ac_dim)
            skill_id: (B, 1)
            skill_horizon: int
        Returns (B, 30) stacked actions by performing the
        skill_id-th skill (refined by given refine_vector) for skill_horizon steps
        """
        stacked_ac = np.empty((ob.shape[0], self._cfg.skill_dim, self._skill_ac_dim))
        for i in range(ob.shape[0]):
            x_pos = ob[i, :3].copy()
            if refine_vector is not None:
                self.refine(refine_vector[i], skill_id[i, 0])
            for j in range(skill_horizon):
                dx_pos = self.skill_actor.skill_ds[skill_id[i, 0]].predict_dx_pos(x_pos)
                # TODO: This is bad. But needed to go through everything without errors
                if np.isnan(dx_pos[0]):
                    Logger.warning("GMM returned NaN. Setting dx to 0.")
                    Logger.warning(
                        f"Skill ID: {skill_id[i, 0]}, RefineVector: {refine_vector[i]}, State: {ob[i, :3]}"
                    )
                    dx_pos = np.zeros_like(dx_pos)
                stacked_ac[i, j] = dx_pos
                x_pos = x_pos + self._cfg.pretrain.dataset.dt * dx_pos
        stacked_ac = torch.from_numpy(
            stacked_ac.reshape(ob.shape[0], self._cfg.skill_dim * self._skill_ac_dim)
        )
        return stacked_ac.to(dtype=self._dtype, device=self._device)

    def preprocess(self, ob, aug=None):
        ob = ob.copy()
        for k, v in ob.items():
            if len(v.shape) >= 4:
                # normalize image to [-0.5, 0.5]
                ob[k] = ob[k] / 255.0 - 0.5
                if aug:
                    ob[k] = aug(ob[k])
            elif self._cfg.env == "maze":
                # normalize state values (position and velocity) to [-0.5, 0.5]
                shape = ob[k].shape
                ob[k] = ob[k].view(-1, shape[-1])
                ob[k] = torch.cat([ob[k][:, :2] / 40 - 0.5, ob[k][:, 2:] / 10], -1)
                ob[k] = ob[k].view(shape)
        return ob


class SeqRefAgent(BaseAgent):
    """SeqRef based on TD-MPC."""

    def __init__(self, cfg, ob_space, ac_space):
        super().__init__(cfg, ob_space)
        self._ob_space = ob_space
        self._ac_space = ac_space
        self._use_amp = cfg.precision == 16
        self._dtype = torch.float16 if self._use_amp else torch.float32
        self._update_iter = 0

        # Meta action space = Skill dim + Refine dim
        meta_ac_space = get_meta_ac_space(cfg)
        # Skill action space = 7
        skill_ac_space = ac_space
        self.ms_agent = SeqRefMetaSkillAgent(
            cfg, ob_space, meta_ac_space, skill_ac_space
        )
        self._aug = RandomShiftsAug()

        if cfg.phase == "rl":
            self.log_alpha = torch.tensor(
                np.log(cfg.alpha_init), device=self._device, requires_grad=True
            )

            if cfg.target_entropy is not None:
                self._target_entropy = cfg.target_entropy
            else:
                self._target_entropy = -gym.spaces.flatdim(meta_ac_space)

        self._build_optims()
        self._build_buffers()
        self._log_creation()

        # Load pretrained model.
        if cfg.phase == "rl" and cfg.pretrain_ckpt_path is not None:
            Logger.warning(f"Load pretrained checkpoint {cfg.pretrain_ckpt_path}")
            ckpt = torch.load(cfg.pretrain_ckpt_path, map_location=self._device)
            ckpt = ckpt["agent"]
            self.load_state_dict(ckpt)

    def get_runner(self, cfg, env, env_eval):
        return SeqRefRolloutRunner(cfg, env, env_eval, self)

    def _build_buffers(self):
        cfg = self._cfg
        self._horizon = cfg.n_skill * cfg.skill_horizon

        # Per-episode replay buffer.
        buffer_keys = ["ob", "ac", "rew", "done", "skill_id"]
        sampler = SeqSampler(cfg.n_skill, sample_last_more=cfg.sample_last_more)
        self.hl_buffer = ReplayBufferEpisode(
            buffer_keys, cfg.buffer_size, sampler.sample_func_one_more_ob, cfg.precision
        )
        self.ms_agent.set_buffer(self.hl_buffer)

        # Load data for pre-training.
        buffer_keys = ["ob", "ac", "done"]
        sampler = SeqSampler(self._horizon + 1)
        self._pretrain_buffer = ReplayBufferEpisode(
            buffer_keys, None, sampler.sample_func_tensor, cfg.precision
        )
        self._pretrain_val_buffer = ReplayBufferEpisode(
            buffer_keys, None, sampler.sample_func_tensor, cfg.precision
        )
        Logger.info("Load pre-training data")
        data = pickle.load(gzip.open(cfg.pretrain.data_path, "rb"))
        data_size = len(data)
        Logger.info(f"Load {data_size} trajectories")
        for i, d in enumerate(data):
            if len(d["obs"]) < len(d["dones"]):
                continue  # Skip incomplete trajectories.
            if cfg.env == "calvin":
                # Only use the first 21 states of non-floating objects.
                d["obs"] = d["obs"][:, :21]
            new_d = dict(ob=d["obs"], ac=d["actions"], done=d["dones"])
            new_d["done"][-1] = 1.0  # Force last step to be done.
            if i < data_size * cfg.pretrain.split.train:
                self._pretrain_buffer.store_episode(new_d, False)
            else:
                self._pretrain_val_buffer.store_episode(new_d, False)

        if cfg.env == "maze":
            self._overlay = imageio.imread("envs/assets/maze_40.png")

    def _log_creation(self):
        Logger.info("Creating a SkiMo agent (TD-MPC)")

    def _build_optims(self):
        cfg = self._cfg
        hl_agent = self.ms_agent
        adam_amp = lambda model, lr: AdamAMP(
            model, lr, cfg.weight_decay, cfg.grad_clip, self._device, self._use_amp
        )
        self.hl_modules = [hl_agent.meta_actor, hl_agent.model, hl_agent.decoder]

        self.hl_model_optim = adam_amp([hl_agent.model, hl_agent.decoder], cfg.model_lr)
        self.hl_actor_optim = adam_amp(hl_agent.meta_actor, cfg.actor_lr)

        if cfg.phase == "rl":
            actor_modules = [hl_agent.meta_actor]
            model_modules = [hl_agent.model, hl_agent.decoder]
            if cfg.sac:
                actor_modules += [hl_agent.model.encoder]
            self.actor_optim = adam_amp(actor_modules, cfg.actor_lr)
            self.model_optim = adam_amp(model_modules, cfg.model_lr)
            self.alpha_optim = adam_amp(self.log_alpha, cfg.alpha_lr)

    def set_step(self, step):
        self.ms_agent.set_step(step)
        self._step = step

    def is_off_policy(self):
        return True

    def store_episode(self, rollouts):
        self.hl_buffer.store_episode(rollouts[0], one_more_ob=True)

    def buffer_state_dict(self):
        return dict(
            hl_buffer=self.hl_buffer.state_dict(),
        )

    def load_buffer_state_dict(self, state_dict):
        self.hl_buffer.append_state_dict(state_dict["hl_buffer"])

    def state_dict(self):
        ret = {
            "ms_agent": self.ms_agent.state_dict(),
            "ob_norm": self._ob_norm.state_dict(),
        }

        ret["hl_model_optim"] = self.hl_model_optim.state_dict()
        ret["hl_actor_optim"] = self.hl_actor_optim.state_dict()

        if self._cfg.phase == "rl":
            ret["actor_optim"] = self.actor_optim.state_dict()
            ret["model_optim"] = self.model_optim.state_dict()
            ret["log_alpha"] = self.log_alpha.cpu().detach().numpy()
            ret["alpha_optim"] = self.alpha_optim.state_dict()
        return ret

    def load_state_dict(self, ckpt):
        self.ms_agent.load_state_dict(ckpt["ms_agent"])
        self.ms_agent.skill_actor.load_params()

        self.hl_model_optim.load_state_dict(ckpt["hl_model_optim"])
        self.hl_actor_optim.load_state_dict(ckpt["hl_actor_optim"])
        optimizer_cuda(self.hl_model_optim, self._device)
        optimizer_cuda(self.hl_actor_optim, self._device)

        if "model_optim" in ckpt:
            self.model_optim.load_state_dict(ckpt["model_optim"])
            self.actor_optim.load_state_dict(ckpt["actor_optim"])
            self.log_alpha.data = torch.tensor(ckpt["log_alpha"], device=self._device)
            self.alpha_optim.load_state_dict(ckpt["alpha_optim"])
            optimizer_cuda(self.model_optim, self._device)
            optimizer_cuda(self.actor_optim, self._device)
            optimizer_cuda(self.alpha_optim, self._device)
        self.to(self._device)

    def preprocess(self, ob, aug=None):
        ob = ob.copy()
        for k, v in ob.items():
            if len(v.shape) >= 4:
                ob[k] = ob[k] / 255.0 - 0.5
                if aug:
                    ob[k] = aug(ob[k])
            elif self._cfg.env == "maze":
                shape = ob[k].shape
                ob[k] = ob[k].view(-1, shape[-1])
                ob[k] = torch.cat([ob[k][:, :2] / 40 - 0.5, ob[k][:, 2:] / 10], -1)
                ob[k] = ob[k].view(shape)
        return ob

    def update(self):
        """Sample a batch from the replay buffer and make one model update in each iteration."""
        train_info = Info()
        sw_data, sw_train = StopWatch(), StopWatch()
        train_iter = self._cfg.train_iter
        if self.warm_up_training():
            self.warm_up_iter = (
                self._cfg.warm_up_step * self._cfg.train_iter // self._cfg.train_every
            )
            train_iter += self.warm_up_iter
        for _ in range(train_iter):
            sw_data.start()
            batch = self.hl_buffer.sample(self._cfg.batch_size)
            sw_data.stop()

            sw_train.start()
            _train_info = self._update_network(batch)
            train_info.add(_train_info)
            sw_train.stop()
        Logger.info(f"Data: {sw_data.average():.3f}  Train: {sw_train.average():.3f}")

        return train_info.get_dict()

    def _update_network(self, batch):
        """Updates skill dynamics model and high-level policy."""
        cfg = self._cfg
        info = Info()
        mse = nn.MSELoss(reduction="none")
        scalars = cfg.scalars
        hl_agent = self.ms_agent
        max_kl = cfg.max_divergence

        if cfg.freeze_model:
            hl_agent.model.encoder.requires_grad_(False)
            hl_agent.model.dynamics.requires_grad_(False)

        # ob: {k: BxTx`ob_dim[k]`}, ac: BxTx`ac_dim` (this is refine vector), rew: BxTx1, skill_id: BxTx1
        o, rv, rew = batch["ob"], batch["ac"], batch["rew"]
        done = batch["done"]
        skill_id = batch["skill_id"]
        o = self.preprocess(o, aug=self._aug)

        # Get refined stacked actions BxT
        ac = self.ms_agent.get_refined_skill_actions(
            o["ob"][:, 0, :].squeeze(1).cpu().numpy(),
            rv.squeeze(1).cpu().numpy(),
            skill_id.cpu().numpy(),
            cfg.skill_horizon,
        )
        ac = ac.unsqueeze(1)

        # Flip dimensions, BxT -> TxB
        def flip(x, l=None):
            if isinstance(x, dict):
                return [{k: v[:, t] for k, v in x.items()} for t in range(l)]
            else:
                return x.transpose(0, 1)

        hl_feat = flip(hl_agent.model.encoder(o))
        # Avoid gradients for the model target
        with torch.no_grad():
            hl_feat_target = flip(hl_agent.model_target.encoder(o))

        ob = flip(o, cfg.n_skill + 1)
        rv = flip(rv)
        ac = flip(ac)
        rew = flip(rew)
        done = flip(done)

        with torch.autocast(cfg.device, enabled=self._use_amp):
            # Trians skill dynamics model.
            z = z_next_pred = hl_feat[0]
            rewards = []

            consistency_loss = 0
            reward_loss = 0
            value_loss = 0
            q_preds = [[], []]
            q_targets = []
            alpha = self.log_alpha.exp().detach()
            for t in range(cfg.n_skill):
                z = z_next_pred
                z_next_pred, reward_pred = hl_agent.model.imagine_step(z, rv[t], ac[t])
                if cfg.sac:
                    z = ob[t]["ob"]
                q_pred = hl_agent.model.critic(z, rv[t])

                with torch.no_grad():
                    # `z` for contrastive learning
                    z_next = hl_feat_target[t + 1]

                    # `z` for `q_target`
                    z_next_q = hl_feat[t + 1]
                    # TODO: Talk to Iman. This is not possible since
                    # we need obs_next for meta action of next state
                    # And what should be the skill_id
                    rv_next_dist = hl_agent.meta_actor(z_next_q)
                    rv_next = rv_next_dist.rsample()
                    if cfg.sac:  # This will be false
                        z_next_q = ob[t + 1]["ob"]
                    q_next = torch.min(*hl_agent.model_target.critic(z_next_q, rv_next))

                    q_target = rew[t] + (1 - done[t].long()) * cfg.rl_discount * q_next
                rewards.append(reward_pred.detach())
                q_preds[0].append(q_pred[0].detach())
                q_preds[1].append(q_pred[1].detach())
                q_targets.append(q_target)

                rho = scalars.rho**t
                consistency_loss += rho * mse(z_next_pred, z_next).mean(dim=1)
                reward_loss += rho * mse(reward_pred, rew[t])
                value_loss += rho * (
                    mse(q_pred[0], q_target) + mse(q_pred[1], q_target)
                )

                # Additional reward prediction loss.
                reward_pred = hl_agent.model.reward(
                    torch.cat([hl_feat[t], rv[t]], dim=-1)
                ).squeeze(-1)
                reward_loss += mse(reward_pred, rew[t])
                # Additional value prediction loss.
                obs = hl_feat[t] if not cfg.sac else ob[t]["ob"]
                q_pred = hl_agent.model.critic(obs, rv[t])
                value_loss += mse(q_pred[0], q_target) + mse(q_pred[1], q_target)

            # If only using SAC, model loss is nothing but the critic loss.
            if cfg.sac:
                consistency_loss *= 0
                reward_loss *= 0

            model_loss = (
                scalars.consistency * consistency_loss.clamp(max=1e5)
                + scalars.hl_reward * reward_loss.clamp(max=1e5) * 0.5
                + scalars.hl_value * value_loss.clamp(max=1e5)
            ).mean()
            model_loss.register_hook(lambda grad: grad * (1 / cfg.n_skill))
        model_grad_norm = self.model_optim.step(model_loss)

        # Trains high-level policy.
        with torch.autocast(cfg.device, enabled=self._use_amp):
            actor_loss = 0
            alpha = self.log_alpha.exp().detach()
            hl_feat = flip(hl_agent.model.encoder(o)) if cfg.sac else hl_feat.detach()
            z = z_next_pred = hl_feat[0]

            # Computes `actor_loss` based on imagined states
            for t in range(cfg.n_skill):
                z = z_next_pred
                r, r_dist = hl_agent.meta_actor.act(z, return_dist=True)
                rho = scalars.rho**t
                if cfg.sac:
                    z = ob[t]["ob"]
                actor_loss += -rho * torch.min(*hl_agent.model.critic(z, r))
                info["actor_std"] = r_dist.base_dist.base_dist.scale.mean().item()

            actor_loss = actor_loss.clamp(-1e5, 1e5).mean()
            actor_loss.register_hook(lambda grad: grad * (1 / cfg.n_skill))
        actor_grad_norm = self.actor_optim.step(actor_loss)

        # Update alpha.
        if cfg.fixed_alpha is None:
            with torch.autocast(cfg.device, enabled=self._use_amp):
                alpha = self.log_alpha.exp()
                # TODO: Talk to Iman about this, double check this, do we also modify policy improvement?
                # Fixed alpha for calvin so no problem but in case we change this in the future
                # Taken from sac_agent.py in rolf
                z = hl_feat[0]
                _, r_dist = hl_agent.meta_actor.act(z, return_dist=True)
                log_pi = r_dist.log_prob(r_dist.rsample())
                alpha_loss = -(alpha * (log_pi + self._target_entropy).detach()).mean()
            self.alpha_optim.step(alpha_loss)

        # Update model target.
        self._update_iter += 1
        if self._update_iter % cfg.target_update_freq == 0:
            soft_copy_network(
                hl_agent.model_target, hl_agent.model, cfg.target_update_tau
            )

        # For logging.
        q_targets = torch.cat(q_targets)
        q_preds[0] = torch.cat(q_preds[0])
        q_preds[1] = torch.cat(q_preds[1])
        info["value_target_min"] = q_targets.min().item()
        info["value_target_max"] = q_targets.max().item()
        info["value_target"] = q_targets.mean().item()
        info["value_predicted_min0"] = q_preds[0].min().item()
        info["value_predicted_min1"] = q_preds[1].min().item()
        info["value_predicted0"] = q_preds[0].mean().item()
        info["value_predicted1"] = q_preds[1].mean().item()
        info["value_predicted_max0"] = q_preds[0].max().item()
        info["value_predicted_max1"] = q_preds[1].max().item()
        info["model_grad_norm"] = model_grad_norm.item()
        info["actor_grad_norm"] = actor_grad_norm.item()
        info["actor_loss"] = actor_loss.mean().item()
        info["model_loss"] = model_loss.mean().item()
        info["consistency_loss"] = consistency_loss.mean().item()
        info["reward_loss"] = reward_loss.mean().item()
        info["critic_loss"] = value_loss.mean().item()
        info["alpha"] = cfg.fixed_alpha or alpha.item()
        info["alpha_loss"] = alpha_loss.item() if cfg.fixed_alpha is None else 0
        rewards = torch.cat(rewards)
        info["reward_predicted"] = rewards.mean().item()
        info["reward_predicted_min"] = rewards.min().item()
        info["reward_predicted_max"] = rewards.max().item()
        info["reward_gt"] = rew.mean().item()
        info["reward_gt_max"] = rew.max().item()
        info["reward_gt_min"] = rew.min().item()

        return info.get_dict()

    def pretrain(self):
        # Pretrain GMM Skills (Load previously trained weights if flagged)
        if self._cfg.pretrain.retrain_gmm:
            self.ms_agent.skill_actor.train(retrain=self._cfg.pretrain.retrain_gmm)
            self._cfg.pretrain.retrain_gmm = False
        # Pretrain Skill Dynamics Model
        train_info = Info()
        sw_data, sw_train = StopWatch(), StopWatch()
        for _ in range(self._cfg.pretrain.train_iter):
            sw_data.start()
            batch = self._pretrain_buffer.sample(self._cfg.pretrain.batch_size)
            sw_data.stop()

            sw_train.start()
            _train_info = self._pretrain(batch)
            train_info.add(_train_info)
            sw_train.stop()
        Logger.info(f"Data: {sw_data.average():.3f}  Train: {sw_train.average():.3f}")

        info = train_info.get_dict()
        Logger.info(
            f"[HL] model loss: {info['hl_model_loss']:.3f}  consistency loss: {info['consistency_loss']:.3f}"
        )
        return info

    def pretrain_eval(self):
        batch = self._pretrain_val_buffer.sample(self._cfg.pretrain.batch_size)
        self._dynamics_visual_eval()
        return self._pretrain(batch, is_train=False)

    def _pretrain(self, batch, is_train=True):
        """Pre-trains skill dynamics model.

        Skill Dynamics:
            Input - State i.e., Robot Obs + Scene Obs, Action i.e., sequentially stacked velocities (needed to go from State to Next State)
            Output - Next State i.e., Robot Obs + Scene Obs
        """
        cfg = self._cfg
        B, H, L = cfg.pretrain.batch_size, cfg.skill_horizon, cfg.n_skill
        scalars = cfg.scalars
        hl_agent = self.ms_agent
        info = Info()
        mse = nn.MSELoss(reduction="none")

        # ob: Bx(LxH+1)x`ob_dim`, ac: Bx(LxH+1)x`ac_dim`
        ob, ac = batch["ob"], batch["ac"]
        o = dict(ob=ob)
        o = self.preprocess(o, aug=self._aug)
        if ac.shape[1] == L * H + 1:
            ac = ac[:, :-1, :3]  # only position

        with torch.autocast(self._cfg.device, enabled=self._use_amp):
            # Trains skill dynamics model.

            def flip(x, l=None):
                """Flip dimensions, BxT -> TxB."""
                if isinstance(x, dict):
                    return [{k: v[:, t] for k, v in x.items()} for t in range(l)]
                else:
                    return x.transpose(0, 1)

            z = torch.clone(ac).view(B, L, -1)  # stacked actions i.e., (B, L, 3*H)
            hl_o = dict(ob=o["ob"][:, ::H])
            hl_feat = flip(hl_agent.model.encoder(hl_o))
            with torch.no_grad():
                hl_feat_target = flip(hl_agent.model_target.encoder(hl_o))
            hl_ac = flip(z)

            # HL observation reconstruction loss.
            hl_ob_pred = hl_agent.decoder(hl_feat)
            hl_recon_losses = {
                k: -hl_ob_pred[k].log_prob(flip(v)).mean() for k, v in hl_o.items()
            }
            hl_recon_loss = sum(hl_recon_losses.values())

            # HL latent state consistency loss.
            h = h_next_pred = hl_feat[0]
            consistency_loss = 0
            hs = [h]
            hl_o = flip(hl_o, L + 1)
            for t in range(L):
                h = h_next_pred
                a = hl_ac[t].detach()
                h_next_pred, _ = hl_agent.model.imagine_step(h, None, a)
                h_next_target = hl_feat_target[t + 1]
                rho = scalars.rho**t
                consistency_loss += rho * mse(h_next_pred, h_next_target).mean(dim=1)
                hs.append(h_next_pred)

            hl_model_loss = (
                scalars.hl_model * hl_recon_loss
                + scalars.consistency * consistency_loss.clamp(max=1e4).mean()
            )
            hl_model_loss.register_hook(lambda grad: grad * (1 / L))

        info["hl_model_loss"] = hl_model_loss.item()
        info["consistency_loss"] = consistency_loss.mean().item()
        info["hl_recon_loss"] = hl_recon_loss.item()

        hl_model_grad_norm = self.hl_model_optim.step(hl_model_loss)
        info["hl_model_grad_norm"] = hl_model_grad_norm.item()
        if is_train:
            self._update_iter += 1
            # Update target networks.
            if self._update_iter % cfg.target_update_freq == 0:
                soft_copy_network(
                    hl_agent.model_target, hl_agent.model, cfg.target_update_tau
                )

        return info.get_dict()

    def _dynamics_visual_eval(self):
        """
        Evaluate the skill dymanics model and its components, visualize predictions and compare them with ground truth
        """
        plot_traj_len = 15
        cfg = self._cfg
        data = pickle.load(gzip.open(cfg.pretrain.data_path, "rb"))
        rand_idx = np.random.randint(0, len(data))
        # Ignore and sample again if chosen radom trajectory is incomplete
        while len(data[rand_idx]["obs"]) < len(data[rand_idx]["dones"]):
            rand_idx = np.random.randint(0, len(data))

        traj = data[rand_idx]
        obs = []
        decoded_obs = []
        with torch.no_grad():
            for i in range(0, len(traj["obs"]) - cfg.skill_horizon, cfg.skill_horizon):
                ob = traj["obs"][i][:21]
                obs.append(ob)
                ac = traj["actions"][i : i + cfg.skill_horizon, :3].reshape(1, -1)
                ac = torch.from_numpy(ac).to(self._device)

                ob_dict = {"ob": torch.from_numpy(ob).unsqueeze(0).to(self._device)}
                z = self.ms_agent.model.encoder(ob_dict)
                next_z, _ = self.ms_agent.model.imagine_step(z, None, ac)

                dec_ob = self.ms_agent.decoder(next_z)  # returns a mixed distribution
                decoded_obs.append(dec_ob["ob"].mode()[0].detach().cpu().numpy())
                if len(decoded_obs) >= plot_traj_len:
                    break

        obs = np.array(obs)
        decoded_obs = np.array(decoded_obs)
        # Plotting
        x_gt, y_gt, z_gt = (
            obs[:plot_traj_len, 0],
            obs[:plot_traj_len, 1],
            obs[:plot_traj_len, 2],
        )
        x_pred, y_pred, z_pred = (
            decoded_obs[:plot_traj_len, 0],
            decoded_obs[:plot_traj_len, 1],
            decoded_obs[:plot_traj_len, 2],
        )

        # Create a new 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the ground truth and predicted trajectories
        ax.plot(x_gt, y_gt, z_gt, color="blue", marker="o", label="Ground Truth")
        ax.plot(
            x_pred, y_pred, z_pred, color="red", marker="o", label="Dynamics Predicted"
        )

        # Add a legend
        ax.legend()

        # Show the plot
        plt.savefig(
            str(
                f"log/temp/{self._update_iter}.png",
            )
        )


def refined_skill_actions(
    cfg, skill_actor, ob, refine_vector, skill_id, skill_horizon, skill_ac_dim
):
    """
    Input
        ob: numpy array (B, 21)
        refine_vector: numpy array (1, self._meta_ac_dim)
        skill_id: (B, 1)
        skill_horizon: int
    Returns numpy array (B, 30) stacked actions by performing the
    """
    stacked_ac = np.empty((ob.shape[0], cfg.skill_dim, skill_ac_dim))
    for i in range(ob.shape[0]):
        x_pos = ob[i, :3]
        if refine_vector[i] is not None:
            refine_dict = get_refine_dict(cfg, refine_vector[i])
            skill_actor.refine_params(refine_dict, skill_id[i, 0])
        for j in range(skill_horizon):
            dx_pos = skill_actor.skill_ds[skill_id[i, 0]].predict_dx_pos(x_pos)
            # TODO: This is bad. But needed to go through everything without errors
            if np.isnan(dx_pos[0]):
                Logger.warning("GMM returned NaN. Setting dx to 0.")
                Logger.warning(
                    f"Skill ID: {skill_id[i, 0]}, RefineVector: {refine_vector[i]}, State: {ob[i, :3]}"
                )
                dx_pos = np.zeros_like(dx_pos)
            stacked_ac[i, j] = dx_pos
            x_pos = x_pos + cfg.pretrain.dataset.dt * dx_pos
        skill_actor.reset_params(skill_id[i, 0])
    return stacked_ac.reshape(ob.shape[0], cfg.skill_dim * skill_ac_dim)
