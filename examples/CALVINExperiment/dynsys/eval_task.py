import os
import sys
import hydra
import wandb
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from examples.CALVINExperiment.envs.task_env import TaskSpecificEnv
from examples.CALVINExperiment.seqblend.CALVINSkill import CALVINSkill

cwd_path = Path(__file__).absolute().parents[0]
calvin_exp_path = cwd_path.parents[0]
root = calvin_exp_path.parents[0]

# This is to access the locally installed repo clone when using slurm
sys.path.insert(0, calvin_exp_path.as_posix())  # CALVINExperiment
sys.path.insert(
    0, os.path.join(calvin_exp_path, "calvin_env")
)  # CALVINExperiment/calvin_env
sys.path.insert(0, root.as_posix())  # Root


class TaskEvaluator(object):
    """Python wrapper that allows you to evaluate learned DS skills
    in the CALVIN environment.
    """

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.task = cfg.target_tasks
        self.task_name = "_".join(self.task)
        self.logger = logging.getLogger("TaskEvaluator")
        self.fixed_ori = None

    def evaluate(self, ds, dataset, max_steps, sampling_dt, render=False, record=False):
        succesful_rollouts, rollout_returns, rollout_lengths = 0, [], []
        rollout_return = 0

        observation = self.env.reset()
        x = self.calibrate_EE_start_state(ds, observation, start_point_sigma=0.05)

        gap = np.linalg.norm(x - ds[0].dataset.start)
        idx = 0
        for step in range(max_steps):
            # Manually switch the DS after some time.
            d_x = ds[idx].predict_dx(x)
            delta_x = sampling_dt * d_x
            new_x = x + delta_x
            if self.cfg.state_type == "pos":
                new_x = np.append(new_x, np.append(self.fixed_ori, -1))
            else:
                new_x = np.append(new_x, -1)
            action = self.env.prepare_action(new_x, type="abs")
            observation, reward, done, info = self.env.step(action)
            x = observation
            rollout_return += reward
            if record:
                self.env.record_frame()
            if render:
                self.env.render()
            # Premature exit when close to skill's goal
            dist_to_goal = np.linalg.norm(observation - ds[idx].goal)
            if dist_to_goal <= 0.01:
                idx = 1
            if done:
                break
            # print(step, delta_x, info['completed_tasks'])
        status = None
        if info["success"]:
            succesful_rollouts += 1
            status = f"Success {rollout_return}"
        else:
            status = f"Fail {rollout_return}"
        self.logger.info(f"{idx+1}: {status}!")
        if record:
            self.logger.info("Saving Robot Camera Obs")
            video_path = self.env.save_recorded_frames(
                self.cfg.exp_dir, f"{np.random.randint(0, 100)}_{status}"
            )
            self.env.reset_recorded_frames()
            if self.cfg.wandb:
                wandb.log(
                    {
                        f"{self.task_name} {status} {self.env.record_count}": wandb.Video(
                            video_path, fps=30, format="gif"
                        )
                    }
                )
        rollout_returns.append(rollout_return)
        rollout_lengths.append(step)
        acc = succesful_rollouts / len(dataset.X)
        if self.cfg.wandb:
            wandb.config.update({"val dataset size": len(dataset.X)})
            wandb.log(
                {
                    "task": self.task_name,
                    "accuracy": acc * 100,
                    "average_return": np.mean(rollout_returns),
                    "average_traj_len": np.mean(rollout_lengths),
                }
            )
        return acc, np.mean(rollout_returns), np.mean(rollout_lengths), status, gap

    def run(self):
        if self.cfg.wandb:
            config = {
                "state_type": self.cfg.state_type,
                "sampling_dt": self.cfg.sampling_dt,
                "max steps": self.cfg.max_steps,
            }
            wandb.init(
                project="ds-evaluation",
                entity="in-ac",
                config=config,
                name=f"{self.task_name}_{self.cfg.state_type}",
            )

        # Create and load models to evaluate
        self.cfg.dim = 3
        ds = [
            CALVINSkill(skill, i, self.cfg.demos_dir, self.cfg.skills_dir)
            for i, skill in enumerate(self.task)
        ]
        self.cfg.skill = self.task[0]
        dataset = hydra.utils.instantiate(self.cfg.dataset)
        self.fixed_ori = dataset.fixed_ori

        self.logger.info(
            f"Evaluating DS of skills in {self.task_name} task with {self.cfg.state_type} input on CALVIN environment"
        )
        # Create and load models to evaluate
        # Evaluate by simulating in the CALVIN environment
        fails = []
        suc = []
        runs = 50
        for _ in range(0, runs):
            acc, avg_return, avg_len, status, gap = self.evaluate(
                ds,
                dataset,
                max_steps=self.cfg.max_steps,
                render=self.cfg.render,
                record=self.cfg.record,
                sampling_dt=self.cfg.dt,
            )
            self.env.count = 0
            if "F" in status:
                fails.append(gap)
            else:
                suc.append(gap)
        if self.cfg.wandb:
            wandb.finish()

        # Log evaluation output
        # self.logger.info(f"{self.task_name} Task Accuracy: {round(acc, 2)}")
        self.logger.info(f"{self.task_name} Fails: {round(sum(fails)/len(fails), 2)}")
        self.logger.info(f"{self.task_name} Succ: {round(sum(suc)/len(suc), 2)}")
        self.logger.info(f"{self.task_name} Fail %: {round(len(fails)/runs, 2)}")
        self.logger.info(f"{self.task_name} Suc %: {round(len(suc)/runs, 2)}")

    def calibrate_EE_start_state(
        self, ds, obs, error_margin=0.01, max_checks=15, start_point_sigma=0.1
    ):
        """Samples a random starting point and moves the end effector to that point"""
        desired_start = ds[0].sample_start(size=1, sigma=start_point_sigma)
        count = 0
        state = np.append(desired_start, np.append(self.fixed_ori, -1))
        action = self.env.prepare_action(state, type="abs")
        while np.linalg.norm(obs - desired_start) > error_margin:
            obs, _, _, _ = self.env.step(action)
            count += 1
            if count >= max_checks:
                # self.cons_logger.info(
                #     f"CALVIN is struggling to place the EE at the right initial pose. \
                #         Difference: {np.linalg.norm(obs - desired_start)}"
                # )
                break
        return obs


@hydra.main(version_base="1.1", config_path="../config", config_name="eval_task")
def main(cfg: DictConfig) -> None:
    new_env_cfg = {**cfg.calvin_env.env}
    new_env_cfg["use_egl"] = False
    new_env_cfg["show_gui"] = False
    new_env_cfg["use_vr"] = False
    new_env_cfg["use_scene_info"] = True
    new_env_cfg["tasks"] = cfg.calvin_env.tasks
    new_env_cfg.pop("_target_", None)
    new_env_cfg.pop("_recursive_", None)
    new_env_cfg["target_tasks"] = cfg.target_tasks
    new_env_cfg["sequential"] = cfg.task_sequential

    env = TaskSpecificEnv(**new_env_cfg)
    env.state_type = cfg.state_type
    cfg.exp_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]

    eval = TaskEvaluator(cfg, env)
    eval.run()


if __name__ == "__main__":
    main()
