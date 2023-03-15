import os
import numpy as np
from SkillsSequencing.skills.mps.dynsys.CALVIN_DS import CALVINDynSysDataset
from SkillsSequencing.skills.mps.dynsys.gmm import ManifoldGMM


class CALVINExp:
    """
    This class defines an experiment for a robot arm with a gripper in CALVIN environment.
    """
    def __init__(self, skill_list):
        """
        Initialization of the experiment class.

        Parameters
        ----------
        :param skill_list: list of skills
        :param skill_fcns: skills functions returning the desired control values for the list of skills

        """
        self.skill_list = skill_list
        self.datasets = self.load_ds_datasets()
        self.skill_ds = self.load_skill_ds()
        self.skill_fncs = [self.get_ds_skill(ds) for ds in self.skill_ds]
        self.desired_values = [None for s in self.skill_list]

    def get_ds_skill(self, ds):
        sampling_dt = 2/30

        def ds_skill(x):
            x = x[:3]
            d_x = ds.predict_dx(x-ds.goal)
            delta_x = sampling_dt * d_x
            x = x + delta_x
            return x

        return ds_skill

    def load_skill_ds(self):
        skills_dir = 'home/lagandua/projects/refining-skill-sequences/examples/CALVINExperiment/skills_ds/'
        state_type = 'pos'
        dim=3
        gmm_components=5
        ds = [ManifoldGMM(n_components=gmm_components) for s in self.skill_list]
        ds_model_dirs = [os.path.join(skills_dir, state_type, s, ds[idx].name) for idx, s in enumerate(self.skill_list)]

        # Works only with GMM DS for now
        for idx in range(len(self.skill_list)):
            ds[idx].start = self.datasets[idx].start
            ds[idx].goal = self.datasets[idx].goal

            ds[idx].skills_dir = ds_model_dirs[idx]
            ds[idx].load_params()
            ds[idx].state_type = state_type
            ds[idx].manifold = ds[idx].make_manifold(dim=dim)

    def load_ds_datasets(self):
        demos_dir='home/lagandua/projects/refining-skill-sequences/examples/CALVINExperiment/data/'
        return [CALVINDynSysDataset(skill=s, train=True, demos_dir=demos_dir, 
                                    state_type='pos', dt=2/30, sampling_dt=2/30,
                                    goal_centered=False, normalized=False) 
                                    for s in self.skill_list]


    def get_taskspace_training_data(self, fname='training.npy'):
        """
        This function returns the training data for a demonstration trajectory. The training data consists of the list
        of skills, the position and velocity along the trajectory, the desired skill values along the trajectory.

        Parameters
        ----------
        :param fname: file containing demonstration trajectory

        Return
        ------
        :return skill_list: list of skills
        :return xt_d: positions of the different skills along the demonstration
        :return dxt_d: velocities of the different skills along the demonstration
        :return desired_: desired control values given by the skills along the trajectory
        """
        demos_dir='home/lagandua/projects/refining-skill-sequences/examples/CALVINExperiment/data/'
        task_dir = '_'.join(self.skill_list)
        demo_log = np.load(os.path.join(demos_dir, task_dir, fname))

        if demo_log is None:
            raise ValueError('Please generate demo first')

        dt = 2/30
        X = demo_log[0]
        dX = (X[2:, :] - X[:-2, :]) / dt
        X = X[1:-1, :]
        for i in range(len(self.skill_list)):
            desired_value = np.stack([self.skill_fcns[i](X[t, :]) for t in np.arange(0, X.shape[0])])
            self.desired_values[i] = desired_value

        Xt = []
        dXt = []
        desired_ = []
        for idx, skill in enumerate(self.skill_list):
            Xt.append(X)
            dXt.append(dX)
            desired_.append(self.desired_values[idx])

        desired_ = np.hstack(desired_)
        Xt_d = np.hstack(Xt)
        dXt_d = np.hstack(dXt)

        return Xt_d, dXt_d, desired_

