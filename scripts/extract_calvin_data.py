import numpy as np
from pathlib import Path
import gzip
import pickle


def angle_between_angles(a, b):
    diff = b - a
    return (diff + np.pi) % (2 * np.pi) - np.pi


def to_relative_action(robot_obs_1, robot_obs_2, max_pos=0.02, max_orn=0.05):
    assert isinstance(robot_obs_1, np.ndarray)
    assert isinstance(robot_obs_2, np.ndarray)

    rel_pos = robot_obs_2[:3] - robot_obs_1[:3]
    rel_pos = rel_pos / max_pos

    rel_orn = angle_between_angles(robot_obs_1[3:6], robot_obs_2[3:6])
    rel_orn = rel_orn / max_orn

    gripper = robot_obs_2[-1]
    return np.concatenate([rel_pos, rel_orn, np.array([gripper])])


root_dir = Path("/home/lagandua/projects/skimo_base/refining-skill-sequences/")
data_dir = root_dir / "data"
calvin_dir = data_dir / "calvin" / "task_D_D"


ids_train = np.load(calvin_dir / "training" / "ep_start_end_ids.npy")
ids_val = np.load(calvin_dir / "validation" / "ep_start_end_ids.npy")
max_episode_len = 500

data = []
for start, end in ids_train:
    f_step_start = start
    for f_step_end in range(
        start + max_episode_len, end + max_episode_len, max_episode_len
    ):
        if f_step_end > end:
            f_step_end = end
        episode_traj = {}
        obs = []
        rgb_gripper = []
        actions = []
        count = 0
        for step in range(f_step_start, f_step_end):
            step_data = np.load(
                calvin_dir / "training" / f"episode_{step:07}.npz", allow_pickle=True
            )
            obs.append(np.concatenate((step_data["robot_obs"], step_data["scene_obs"])))
            rgb_gripper.append(step_data["rgb_gripper"])
            actions.append(
                to_relative_action(step_data["robot_obs"][:7], step_data["actions"][:7])
            )
        episode_traj["obs"] = np.array(obs)
        episode_traj["rgb_gripper"] = np.array(rgb_gripper)
        episode_traj["actions"] = np.array(actions)
        episode_traj["dones"] = np.zeros(len(obs))
        data.append(episode_traj)
        f_step_start = f_step_end

for start, end in ids_val:
    f_step_start = start
    for f_step_end in range(
        start + max_episode_len, end + max_episode_len, max_episode_len
    ):
        if f_step_end > end:
            f_step_end = end
        episode_traj = {}
        obs = []
        rgb_gripper = []
        actions = []
        for step in range(f_step_start, f_step_end):
            step_data = np.load(
                calvin_dir / "validation" / f"episode_{step:07}.npz", allow_pickle=True
            )
            obs.append(np.concatenate((step_data["robot_obs"], step_data["scene_obs"])))
            rgb_gripper.append(step_data["rgb_gripper"])
            actions.append(
                to_relative_action(step_data["robot_obs"][:7], step_data["actions"][:7])
            )
        episode_traj["obs"] = np.array(obs)
        episode_traj["rgb_gripper"] = np.array(rgb_gripper)
        episode_traj["actions"] = np.array(actions)
        episode_traj["dones"] = np.zeros(len(obs))
        data.append(episode_traj)
        f_step_start = f_step_end

print(f"Found {len(data)} trajectories in total.")
pickle.dump(data, gzip.open(data_dir / "calvin_w_gripper.gz", "wb"))
