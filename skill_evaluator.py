import os
import hydra
import logging
from omegaconf import DictConfig
import time
import numpy as np
import gym
import imageio

from rolf.utils import make_env
from lib.dynsys.gmm_actor import GMMSkillActor


class TaskEval(object):
    """
    Python wrapper that allows you to train DS skills on a given dataset
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.env = make_env(cfg.env.id, cfg.env, cfg.seed, cfg.rolf.name)
        self.env_ac_space = self.env.action_space
        self.skills = self.cfg.rolf.pretrain.skills
        self.env.env.env.env.target_tasks = np.copy(self.skills)
        self.env.env.env.env.tasks_to_complete = np.copy(self.skills)
        self.task_name = "_".join(self.skills)
        self.actor = GMMSkillActor(self.cfg.rolf.pretrain)
        self.actor.load_params()
        self.state_type = self.cfg.rolf.pretrain.dataset.input_type

        # Logging
        self.logs_out_dir = os.path.join(
            "log/task-eval", time.strftime("%m-%d-%H%M%S", time.gmtime(time.time()))
        )
        os.makedirs(self.logs_out_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(self.logs_out_dir, "logs.log"))
        fh.setLevel(logging.DEBUG)
        self.logger = logging.getLogger("TaskEval")
        self.logger.addHandler(fh)
        self.frames = []

    def act(self, obs, skill_id):
        dx, dx_ori = self.actor.act(obs["ob"], skill_id)
        env_ac = np.append(
            dx, np.append(dx_ori, -1)
        )  # pos, ori, gripper_width i.e., size=7
        env_ac = gym.spaces.unflatten(self.env_ac_space, env_ac)
        return env_ac

    def run(self):
        succesful_rollouts, rollout_returns, rollout_lengths = 0, [], []
        rollout_return = 0
        skill_id = 0
        num_evals = 100
        for eval_no in range(num_evals):
            obs = self.env.reset()
            if self.cfg.rolf.record:
                self.record_frame()
            if self.cfg.rolf.render:
                self.env.render()
            skill_id = 0
            for step in range(self.env.max_episode_steps):
                action = self.act(obs, skill_id)
                obs, reward, done, info = self.env.step(action)
                rollout_return += reward

                if self.cfg.rolf.record:
                    self.record_frame()
                if self.cfg.rolf.render:
                    self.env.render()

                if reward > 0:
                    skill_id += 1
                    if skill_id == 4:
                        done = True

                if done:
                    break
            status = None
            if info["success"]:
                succesful_rollouts += 1
                status = "Success"
            else:
                status = "Fail"

            self.logger.info(
                f"{eval_no}: Skills Executed: {skill_id+1} Status: {status} Reward: {rollout_return}!"
            )
            rollout_returns.append(rollout_return)
            rollout_lengths.append(step)
            rollout_return = 0

            if self.cfg.rolf.record:
                self.save_recording(eval_no)
                self.frames = []

        acc = succesful_rollouts / num_evals
        self.logger.info(f"Accuracy: {acc * 100}")
        self.logger.info(f"Avg. Return: {np.mean(rollout_returns)}")
        self.logger.info(f"Avg. Steps: {np.mean(rollout_lengths)}")
        self.logger.info(f"Task: {self.task_name}")

    def record_frame(self, obs_type="rgb", cam_type="static", size=200):
        """Record RGB obsservations"""
        rgb_obs, depth_obs = self.env.get_camera_obs()
        if obs_type == "rgb":
            frame = rgb_obs[f"{obs_type}_{cam_type}"]
        else:
            frame = depth_obs[f"{obs_type}_{cam_type}"]
        self.frames.append(frame)

    def save_recording(self, fname):
        """Save recorded frames as a video"""
        if len(self.frames) == 0:
            # This shouldn't happen but if it does, the function
            # call exits gracefully
            return None
        fname = f"{fname}.gif"
        kargs = {"duration": 33}
        fpath = os.path.join(self.logs_out_dir, fname)
        imageio.mimsave(fpath, np.array(self.frames), "GIF", **kargs)
        return fpath


@hydra.main(config_path="config", config_name="seqref_calvin")
def main(cfg: DictConfig) -> None:
    eval = TaskEval(cfg)
    eval.run()


if __name__ == "__main__":
    main()
