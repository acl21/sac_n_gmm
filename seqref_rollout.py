"""
Runs rollouts (RolloutRunner class) and collects transitions using Rollout class.
"""

import numpy as np
import gym.spaces

from rolf.utils import Logger, Info, Every
from rolf.algorithms.rollout import Rollout, RolloutRunner


class SeqRefRolloutRunner(RolloutRunner):
    """Rollout hierarchical policy."""

    def __init__(self, cfg, env, env_eval, agent):
        """
        Args:
            cfg: configurations for the environment.
            env: training environment.
            env_eval: testing environment.
            agent: policy.
        """
        self._cfg = cfg
        self._env = env
        self._env_eval = env_eval
        self._agent = agent
        self._ms_agent = agent.ms_agent
        self._exclude_rollout_log = ["episode_success_state"]

    def run(
        self,
        every_steps=None,
        every_episodes=None,
        log_prefix="",
        step=0,
    ):
        """
        Collects trajectories and yield every `every_steps`/`every_episodes`.

        Args:
            every_steps: if not None, returns rollouts `every_steps`.
            every_episodes: if not None, returns rollouts `every_epiosdes`.
            log_prefix: log as `log_prefix` rollout: %s.
        """
        if every_steps is None and every_episodes is None:
            raise ValueError("Both every_steps and every_episodes cannot be None")

        cfg = self._cfg
        env = self._env
        ms_agent = self._ms_agent

        done_rollout = (
            Every(every_steps, step) if every_steps else Every(every_episodes, 0)
        )

        # Initialize rollout buffer.
        meta_rollout = Rollout(
            ["ob", "ac", "rew", "done", "skill_id"], cfg.rolf.precision
        )
        rollout = Rollout(
            ["ob", "meta_ac", "ac", "rew", "done", "skill_id"], cfg.rolf.precision
        )
        reward_info = Info()
        ep_info = Info()
        episode = 0
        rollout_len = 0
        skill_ids = np.concatenate(
            [
                np.repeat(x, env.max_episode_steps / len(env.target_tasks))
                for x in range(len(env.target_tasks))
            ]
        )
        skill_id = 0
        is_goal_reached = False

        while True:
            done = False
            ep_len = 0
            ep_rew = 0
            ob_next = env.reset()
            ob = ob_next
            state_next_hl = None
            skill_id = 0
            is_goal_reached = False

            rollout.add(dict(ob=ob_next))
            meta_rollout.add(dict(ob=ob_next))

            # Rollout one episode.
            while not done and skill_id < 4:
                state_hl = state_next_hl

                # Sample meta (high-level) action from meta policy.
                if step < cfg.rolf.warm_up_step:
                    meta_ac, state_next_hl = ms_agent.meta_act(
                        ob_next, skill_id, state_hl, is_train=True, warmup=True
                    )
                else:
                    meta_ac, state_next_hl = ms_agent.meta_act(
                        ob_next, skill_id, state_hl, is_train=True
                    )
                # Add skill_id now as it may be updated later
                meta_rollout.add(dict(skill_id=skill_id))

                # Update skill parameters with meta actions (refine GMM here)
                ms_agent.refine(meta_ac, skill_id)

                # Execute a skill.
                skill_len = 0
                meta_rew = 0
                while not done and skill_len < cfg.rolf.skill_horizon:
                    # Break if goal state of the skill_id-th skill is reached
                    if is_goal_reached:
                        skill_id += 1  # TODO: Talk to Iman about this
                        skill_len = cfg.rolf.skill_horizon
                        break

                    ob = ob_next

                    # Sample low-level action from skill policy.
                    ac, is_goal_reached = ms_agent.skill_act(
                        ob["ob"], None, skill_id, is_train=False
                    )

                    # Take a step.
                    ob_next, reward, done, info = env.step(ac)
                    info.update(env.get_episode_info())
                    info.update(dict(meta_ac=meta_ac))

                    step += 1
                    ep_len += 1
                    ep_rew += reward
                    skill_len += 1
                    meta_rew += reward

                    flat_ac = gym.spaces.flatten(env.action_space, ac)
                    rollout.add(dict(ob=ob_next, ac=flat_ac, done=done))
                    rollout.add(dict(meta_ac=meta_ac))
                    rollout.add(dict(rew=reward * cfg.rolf.reward_scale))
                    rollout.add(dict(skill_id=skill_id))
                    reward_info.add(info)
                    rollout_len += 1

                meta_rollout.add(dict(ob=ob_next, ac=meta_ac, done=done))
                meta_rollout.add(dict(rew=meta_rew * cfg.rolf.reward_scale))
                if every_steps and done_rollout(step):
                    yield (
                        meta_rollout.get(),
                        rollout.get(),
                    ), rollout_len, ep_info.get_dict(reduction="max", only_scalar=True)
                    rollout_len = 0

            # Compute average/sum/max of information.
            reward_info_dict = reward_info.get_dict(reduction="max", only_scalar=True)
            reward_info_dict.update(
                dict(len=ep_len, rew=ep_rew * cfg.rolf.reward_scale)
            )
            ep_info.add(reward_info_dict)

            Logger.info(
                "rollout: %s",
                {
                    k: v
                    for k, v in reward_info_dict.items()
                    if k not in self._exclude_rollout_log and np.isscalar(v)
                },
            )

            episode += 1
            if every_episodes and done_rollout(episode):
                yield (
                    meta_rollout.get(),
                    rollout.get(),
                ), rollout_len, ep_info.get_dict(only_scalar=True)
                rollout_len = 0

    def run_episode(self, record_video=False):
        """
        Runs one episode and returns the rollout (mainly for evaluation).

        Args:
            record_video: record video of rollout if True.
        """
        cfg = self._cfg
        env = self._env_eval
        ms_agent = self._ms_agent

        # Initialize rollout buffer.
        rollout = Rollout(
            ["ob", "meta_ac", "ac", "rew", "done", "skill_id"], cfg.rolf.precision
        )
        reward_info = Info()

        done = False
        ep_len = 0
        ep_rew = 0
        ob_next = env.reset()
        ob = ob_next
        state_next_hl = None
        skill_ids = np.concatenate(
            [
                np.repeat(x, env.max_episode_steps / len(env.target_tasks))
                for x in range(len(env.target_tasks))
            ]
        )
        skill_id = 0
        is_goal_reached = False

        record_frames = []
        if record_video:
            record_frames.append(self._render_frame(ep_len, ep_rew))

        # Rollout one episode.
        while not done and skill_id < 4:
            state_hl = state_next_hl

            # Sample meta action from meta policy.
            meta_ac, state_next_hl = ms_agent.meta_act(
                ob_next, state_hl, is_train=False
            )

            # Update skill parameters with meta actions (refine GMM here)
            ms_agent.refine(meta_ac, skill_id)

            skill_len = 0
            while not done and skill_len < cfg.rolf.skill_horizon:
                # Break if goal state of the skill_id-th skill is reached
                if is_goal_reached:
                    skill_id += 1  # TODO: Talk to Iman about this
                    skill_len = cfg.rolf.skill_horizon
                    break

                ob = ob_next

                # Sample low-level action from skill policy.
                ac, is_goal_reached = ms_agent.skill_act(
                    ob["ob"], None, skill_id, is_train=False
                )

                # Take a step.
                ob_next, reward, done, info = env.step(ac)
                info.update(env.get_episode_info())
                info.update(dict(meta_ac=meta_ac))

                ep_len += 1
                ep_rew += reward
                skill_len += 1

                flat_ac = gym.spaces.flatten(env.action_space, ac)
                rollout.add(dict(ob=ob_next, ac=flat_ac, done=done))
                rollout.add(dict(meta_ac=meta_ac, rew=reward * cfg.rolf.reward_scale))
                rollout.add(dict(skill_id=skill_id))
                reward_info.add(info)

                if record_video:
                    frame_info = info.copy()
                    record_frames.append(self._render_frame(ep_len, ep_rew, frame_info))

        # Compute average/sum/max of information.
        ep_info = {"len": ep_len, "rew": ep_rew}
        if "episode_success_state" in reward_info.keys():
            ep_info["episode_success_state"] = reward_info["episode_success_state"]
        ep_info.update(reward_info.get_dict(reduction="max", only_scalar=True))

        Logger.info(
            "rollout: %s",
            {
                k: v
                for k, v in ep_info.items()
                if k not in self._exclude_rollout_log and np.isscalar(v)
            },
        )
        return rollout.get(), ep_info, record_frames
