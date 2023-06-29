import pickle
import gzip

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

from seqref_rollout import SeqRefRolloutRunner


class SeqRefMetaAgent(BaseAgent):
    """High-level agent for SeqRef."""

    def __init__(self, cfg, ob_space, ac_space):
        super().__init__(cfg, ob_space)
        self._ob_space = ob_space
        self._ac_space = ac_space
        self._use_amp = cfg.precision == 16
        self._dtype = torch.float16 if self._use_amp else torch.float32
        self._std_decay = LinearDecay(cfg.max_std, cfg.min_std, cfg.std_step)
        self._horizon_decay = LinearDecay(1, cfg.n_skill, cfg.horizon_step)
        self._update_iter = 0
        self._ac_dim = cfg.skill_dim
        self._ob_dim = gym.spaces.flatdim(ob_space)

        self.model = TDMPCModel(cfg, self._ob_space, cfg.skill_dim, self._dtype)
        self.model_target = TDMPCModel(cfg, self._ob_space, cfg.skill_dim, self._dtype)
        copy_network(self.model_target, self.model)
        self.actor = ActionDecoder(
            cfg.state_dim,
            cfg.skill_dim,
            [cfg.num_units] * cfg.num_layers,
            cfg.dense_act,
            cfg.log_std,
        )
        self.decoder = Decoder(cfg.decoder, cfg.state_dim, self._ob_space)
        self.to(self._device)

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "model": self.model.state_dict(),
            "model_target": self.model_target.state_dict(),
            "decoder": self.decoder.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self.actor.load_state_dict(ckpt["actor"])
        self.model.load_state_dict(ckpt["model"])
        self.model_target.load_state_dict(ckpt["model_target"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.to(self._device)

    @property
    def ac_space(self):
        return self._ac_space

    @torch.no_grad()
    def estimate_value(self, state, ac, horizon):
        """Imagine a trajectory for `horizon` steps, and estimate the value."""
        value, discount = 0, 1
        for t in range(horizon):
            state, reward = self.model.imagine_step(state, ac[t])
            value += discount * reward
            discount *= self._cfg.rl_discount
        value += discount * torch.min(*self.model.critic(state, self.actor.act(state)))
        return value

    @torch.no_grad()
    def plan(self, ob, prev_mean=None, is_train=True):
        """Plan given an observation `ob`."""
        pass

    @torch.no_grad()
    def act(self, ob, mean=None, is_train=True, warmup=False):
        """Returns action and the actor's activation given an observation `ob`."""
        pass

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


class SeqRefSkillAgent(BaseAgent):
    """
    Low-level agent for SeqRef.
    - Learns GMMs for all skills
    - Learns the skill dynamic model on task agnostic data
    """

    def __init__(self, cfg, ob_space, ac_space):
        super().__init__(cfg, ob_space)
        self._ob_space = ob_space
        self._ac_space = ac_space
        self._use_amp = cfg.precision == 16
        self._dtype = torch.float16 if self._use_amp else torch.float32
        self._update_iter = 0

        self._ac_dim = ac_dim = gym.spaces.flatdim(self._ac_space)
        hidden_dims = [cfg.num_units] * cfg.num_layers
        self.encoder = Encoder(cfg.encoder, ob_space, cfg.state_dim)

        if cfg.lstm:
            skill_input_dim = ac_dim + cfg.state_dim
            self.skill_encoder = LSTMEncoder(
                skill_input_dim, cfg.skill_dim, cfg.lstm_units, 1, log_std=cfg.log_std
            )
        else:
            skill_input_dim = cfg.skill_horizon * ac_dim + cfg.state_dim
            self.skill_encoder = DenseDecoderTanh(
                skill_input_dim, cfg.skill_dim, hidden_dims, cfg.dense_act, cfg.log_std
            )
        self.actor = ActionDecoder(
            cfg.state_dim + cfg.skill_dim,
            ac_dim,
            hidden_dims,
            cfg.dense_act,
            cfg.log_std,
        )
        self.to(self._device)

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "encoder": self.encoder.state_dict(),
            "skill_encoder": self.skill_encoder.state_dict(),
            "ob_norm": self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self.actor.load_state_dict(ckpt["actor"])
        self.encoder.load_state_dict(ckpt["encoder"])
        self.skill_encoder.load_state_dict(ckpt["skill_encoder"])
        self.to(self._device)

    @property
    def ac_space(self):
        return self._ac_space

    @torch.no_grad()
    def act(self, ob, mean=None, cond=None, is_train=True):
        """Returns action and the actor's activation given an observation `ob`."""
        ob = ob.copy()
        for k, v in ob.items():
            ob[k] = np.expand_dims(v, axis=0).copy()

        self.encoder.eval()
        self.actor.eval()
        with torch.autocast(self._cfg.device, enabled=self._use_amp):
            ob = to_tensor(ob, self._device, self._dtype)
            ob = self.preprocess(ob)
            feat = self.encoder(ob)
            if cond is not None:
                cond = to_tensor(cond, self._device, self._dtype).unsqueeze(0)
            ac = self.actor.act(feat, cond, deterministic=True)
            ac = ac.cpu().numpy().squeeze(0)
            ac = gym.spaces.unflatten(self._ac_space, ac)
        self.encoder.train()
        self.actor.train()
        return ac, mean

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


class SeqRefAgent(BaseAgent):
    """SeqRef based on TD-MPC."""

    def __init__(self, cfg, ob_space, ac_space):
        super().__init__(cfg, ob_space)
        self._ob_space = ob_space
        self._ac_space = ac_space
        self._use_amp = cfg.precision == 16
        self._dtype = torch.float16 if self._use_amp else torch.float32
        self._update_iter = 0

        meta_ac_space = gym.spaces.Box(-1, 1, [cfg.skill_dim])
        self.meta_agent = SeqRefMetaAgent(cfg, ob_space, meta_ac_space)
        self.skill_agent = SeqRefSkillAgent(cfg, ob_space, ac_space)
        self._aug = RandomShiftsAug()

        self._build_optims()
        self._build_buffers()
        self._log_creation()

        # Load pretrained model.
        if cfg.phase == "rl" and cfg.pretrain_ckpt_path is not None:
            Logger.warning(f"Load pretrained checkpoint {cfg.pretrain_ckpt_path}")
            ckpt = torch.load(cfg.pretrain_ckpt_path, map_location=self._device)
            ckpt = ckpt["agent"]
            ckpt["skill_prior"] = ckpt["meta_agent"].copy()
            self.load_state_dict(ckpt)

    def get_runner(self, cfg, env, env_eval):
        return SeqRefRolloutRunner(cfg, env, env_eval, self)

    def _build_buffers(self):
        cfg = self._cfg
        self._horizon = cfg.n_skill * cfg.skill_horizon

        # Per-episode replay buffer.
        buffer_keys = ["ob", "ac", "rew", "done"]
        sampler = SeqSampler(cfg.n_skill, sample_last_more=cfg.sample_last_more)
        self.hl_buffer = ReplayBufferEpisode(
            buffer_keys, cfg.buffer_size, sampler.sample_func_one_more_ob, cfg.precision
        )
        self.meta_agent.set_buffer(self.hl_buffer)

        # Load data for pre-training of latent dynamics model.
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

    def _log_creation(self):
        Logger.info("Creating a SeqRef agent (TD-MPC)")

    def _build_optims(self):
        cfg = self._cfg
        hl_agent = self.meta_agent
        adam_amp = lambda model, lr: AdamAMP(
            model, lr, cfg.weight_decay, cfg.grad_clip, self._device, self._use_amp
        )
        self.hl_modules = [hl_agent.actor, hl_agent.model, hl_agent.decoder]

        # Optimize the skill dynamics and skills jointly.
        self.hl_model_optim = adam_amp([hl_agent.model, hl_agent.decoder], cfg.model_lr)
        self.hl_actor_optim = adam_amp(hl_agent.actor, cfg.actor_lr)

        if cfg.phase == "rl":
            actor_modules = [hl_agent.actor]
            model_modules = [hl_agent.model, hl_agent.decoder]
            if cfg.sac:
                actor_modules += [hl_agent.model.encoder]
            self.actor_optim = adam_amp(actor_modules, cfg.actor_lr)
            self.model_optim = adam_amp(model_modules, cfg.model_lr)
            self.alpha_optim = adam_amp(self.log_alpha, cfg.alpha_lr)
