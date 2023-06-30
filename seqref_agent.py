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

from lib.dynsys.manifold_gmm_agent import ManifoldGMMSkillAgent
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

        torch.set_default_dtype(self._dtype)
        self.model = TDMPCModel(cfg, self._ob_space, 3 * cfg.skill_dim, self._dtype)
        self.model_target = TDMPCModel(
            cfg, self._ob_space, 3 * cfg.skill_dim, self._dtype
        )
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
        # CALVIN Skill like class to manage DS input and output
        self.agent = ManifoldGMMSkillAgent(cfg.pretrain)
        self.to(self._device)

    def train(self):
        self.agent.train()

    def load_pretrained_weights(self):
        self.agent.load_params()

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "encoder": self.encoder.state_dict(),
            "skill_encoder": self.skill_encoder.state_dict(),
            "ob_norm": self._ob_norm.state_dict(),
        }

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

        if cfg.pretrain.retrain_gmm:
            self.skill_agent.train()
        else:
            self.skill_agent.load_pretrained_weights()

        if cfg.phase == "rl":
            self.log_alpha = torch.tensor(
                np.log(cfg.alpha_init), device=self._device, requires_grad=True
            )

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
        buffer_keys = ["ob", "ac", "rew", "done"]
        sampler = SeqSampler(cfg.n_skill, sample_last_more=cfg.sample_last_more)
        self.hl_buffer = ReplayBufferEpisode(
            buffer_keys, cfg.buffer_size, sampler.sample_func_one_more_ob, cfg.precision
        )
        self.meta_agent.set_buffer(self.hl_buffer)

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
        hl_agent = self.meta_agent
        adam_amp = lambda model, lr: AdamAMP(
            model, lr, cfg.weight_decay, cfg.grad_clip, self._device, self._use_amp
        )
        self.hl_modules = [hl_agent.actor, hl_agent.model, hl_agent.decoder]

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

    def set_step(self, step):
        self.meta_agent.set_step(step)
        self.skill_agent.set_step(step)
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
            "meta_agent": self.meta_agent.state_dict(),
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
        self.meta_agent.load_state_dict(ckpt["meta_agent"])

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

    def pretrain(self):
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
        return self._pretrain(batch, is_train=False)

    def _pretrain(self, batch, is_train=True):
        """Pre-trains skills, skill dynamics model, and skill prior."""
        cfg = self._cfg
        B, H, L = cfg.pretrain.batch_size, cfg.skill_horizon, cfg.n_skill
        scalars = cfg.scalars
        hl_agent = self.meta_agent
        info = Info()
        mse = nn.MSELoss(reduction="none")

        # ob: Bx(LxH+1)x`ob_dim`, ac: Bx(LxH+1)x`ac_dim`
        ob, ac = batch["ob"], batch["ac"]
        o = dict(ob=ob)
        o = self.preprocess(o, aug=self._aug)
        if ac.shape[1] == L * H + 1:
            ac = ac[:, :-1, :3]  # only position

        with torch.autocast(self._cfg.device, enabled=self._use_amp):
            # Trains skill dynamics model and skill prior.

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
                h_next_pred, _ = hl_agent.model.imagine_step(h, a)
                h_next_target = hl_feat_target[t + 1]
                rho = scalars.rho**t
                consistency_loss += rho * mse(h_next_pred, h_next_target).mean(dim=1)
                hs.append(h_next_pred)

            hl_model_loss = (
                scalars.hl_model * hl_recon_loss
                + scalars.consistency * consistency_loss.clamp(max=1e4).mean()
            )
            hl_model_loss.register_hook(lambda grad: grad * (1 / L))

            hl_loss = hl_model_loss

        info["hl_loss"] = hl_loss.item()
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
