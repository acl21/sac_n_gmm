import os
import numpy as np
from examples.CALVINExperiment.CALVINSkill import CALVINSkill
from SkillsSequencing.skills.mps.dynsys.CALVIN_DS import CALVINDynSysDataset
from SkillsSequencing.skills.mps.dynsys.gmm import ManifoldGMM


class CALVINExp:
    """
    This class defines an experiment for a robot arm with a gripper in CALVIN environment.
    """
    def __init__(self, skill_names):
        """
        Initialization of the experiment class.

        Parameters
        ----------
        :param skill_list: list of skills
        :param skill_fcns: skills functions returning the desired control values for the list of skills

        """
        self.skill_names = skill_names
        self.skill_list = [CALVINSkill(skill_name=skill) for skill in self.skill_names]
        self.skill_fncs = [self.get_ds_skill(skill) for skill in self.skill_list]

    def get_ds_skill(self, skill):
        sampling_dt = 2/30

        def ds_skill(x):
            x = x[:3]
            d_x = skill.predict_dx(x)
            delta_x = sampling_dt * d_x
            x = x + delta_x
            return x

        return ds_skill

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
        demos_dir='E:/Uni-Freiburg/Research/refining-skill-sequences/examples/CALVINExperiment/data/'
        task_dir = '_'.join(self.skill_names)
        demo_log = np.load(os.path.join(demos_dir, task_dir, fname))

        if demo_log is None:
            raise ValueError('Please generate demo first')
        else:
            print(f'Loaded task demo data from {os.path.join(demos_dir, task_dir, fname)}')

        dt = 2/30
        X = demo_log[0]
        dX = (X[2:, :] - X[:-2, :]) / dt
        X = X[1:-1, :]
        for i in range(len(self.skill_list)):
            desired_value = np.stack([self.skill_fncs[i](X[t, :]) for t in np.arange(0, X.shape[0])])
            self.skill_list[i].desired_value = desired_value

        Xt = []
        dXt = []
        desired_ = []
        for i in range(len(self.skill_list)):
            Xt.append(X)
            dXt.append(dX)
            desired_.append(self.skill_list[i].desired_value)

        desired_ = np.hstack(desired_)
        Xt_d = np.hstack(Xt)
        dXt_d = np.hstack(dXt)

        print(f'Returning task space training data!')
        return Xt_d, dXt_d, desired_

if __name__ == "__main__":
    exp = CALVINExp(skill_names=['open_drawer', 'close_drawer'])
    Xt_d, dXt_d, desired_ = exp.get_taskspace_training_data()