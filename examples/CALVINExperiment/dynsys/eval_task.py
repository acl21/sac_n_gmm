import os
import sys
import hydra
import wandb
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from examples.CALVINExperiment.envs.task_env import TaskSpecificEnv

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

    def evaluate(self, ds, dataset, max_steps, sampling_dt, render=False, record=False):
        succesful_rollouts, rollout_returns, rollout_lengths = 0, [], []
        start_idx, end_idx = 0, 3
        rollout_return = 0

        x0 = ds[0].start
        if self.cfg.state_type == "pos":
            x0 = np.append(x0, np.append(dataset.fixed_ori, -1))
        else:
            x0 = np.append(x0, -1)
        action = self.env.prepare_action(x0, type="abs")

        observation = self.env.reset()
        current_state = observation[start_idx : end_idx + end_idx]
        count = 0
        error_margin = 0.01
        while np.linalg.norm(current_state - x0[:-1]) >= error_margin:
            observation, reward, done, info = self.env.step(action)
            current_state = observation[start_idx : end_idx + end_idx]
            count += 1
            if count >= 200:
                self.logger.info(
                    "CALVIN is struggling to place the EE at the right initial pose"
                )
                self.logger.info(x0, current_state, np.linalg.norm(current_state - x0))
                break
        # self.logger.info(f'Simulating with DS')
        if record:
            self.logger.info(f"Recording Robot Camera Obs")
            self.env.record_frame()
        idx = 0
        x = current_state[start_idx:end_idx]
        for step in range(max_steps):
            # Manually switch the DS after some time.
            if step == 62:
                idx += 1

            d_x = ds[idx].predict_dx(x - ds[idx].goal)
            delta_x = sampling_dt * d_x
            new_x = x + delta_x
            if self.cfg.state_type == "pos":
                new_x = np.append(new_x, np.append(dataset.fixed_ori, -1))
            else:
                new_x = np.append(new_x, -1)
            action = self.env.prepare_action(new_x, type="abs")
            observation, reward, done, info = self.env.step(action)
            x = observation[start_idx:end_idx]
            rollout_return += reward
            if record:
                self.env.record_frame()
            if render:
                self.env.render()
            if done:
                break
            # print(step, delta_x, info['completed_tasks'])
        status = None
        if info["success"]:
            succesful_rollouts += 1
            status = "Success"
        else:
            status = "Fail"
        self.logger.info(f"{idx+1}: {status}!")
        if record:
            self.logger.info(f"Saving Robot Camera Obs")
            video_path = self.env.save_recorded_frames(self.cfg.exp_dir, f"{status}")
            self.env.reset_recorded_frames()
            status = None
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
        import pdb

        pdb.set_trace()
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
        return acc, np.mean(rollout_returns), np.mean(rollout_lengths)

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
        ds = [hydra.utils.instantiate(self.cfg.dyn_sys) for t in self.task]
        ds_model_dirs = [
            os.path.join(self.cfg.skills_dir, self.cfg.state_type, t, ds[idx].name)
            for idx, t in enumerate(self.task)
        ]

        dataset = None
        # Works only with GMM DS for now
        for idx in range(len(self.task)):
            self.cfg.dataset.train = True
            self.cfg.dataset.skill = self.task[idx]
            dataset = hydra.utils.instantiate(self.cfg.dataset)
            ds[idx].start = dataset.start
            ds[idx].goal = dataset.goal

            ds[idx].skills_dir = ds_model_dirs[idx]
            ds[idx].load_params()
            ds[idx].state_type = self.cfg.state_type
            ds[idx].manifold = ds[idx].make_manifold(self.cfg.dim)

        self.logger.info(
            f"Evaluating DS of skills in {self.task_name} task with {self.cfg.state_type} input on CALVIN environment"
        )
        # Create and load models to evaluate
        # Evaluate by simulating in the CALVIN environment
        acc, avg_return, avg_len = self.evaluate(
            ds,
            dataset,
            max_steps=self.cfg.max_steps,
            render=self.cfg.render,
            record=self.cfg.record,
            sampling_dt=self.cfg.sampling_dt,
        )
        self.env.count = 0
        if self.cfg.wandb:
            wandb.finish()

        # Log evaluation output
        self.logger.info(f"{self.task_name} Task Accuracy: {round(acc, 2)}")


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
