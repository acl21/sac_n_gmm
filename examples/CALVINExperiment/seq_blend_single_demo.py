import os
import inspect
import numpy as np

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
main_dir = current_dir + '/../../'
os.sys.path.insert(0, '../' + current_dir)
os.sys.path.insert(0, main_dir)

import sys
from optparse import OptionParser
from torch.utils.data import DataLoader

from examples.CALVINExperiment.CALVINExp import CALVINExp
from SkillsSequencing.qpnet import qpnet_policies as policy_classes
from SkillsSequencing.qpnet.spec_datasets import SkillDataset
from SkillsSequencing.utils.utils import prepare_torch

device = prepare_torch()

def CALVIN_experiment(options):
    learning_rate = 0.1
    MAX_EPOCH = 2000
    model_fpath = main_dir + '/' + options.policy_file
    data_file = main_dir + '/' + options.data_file

    if os.path.exists(model_fpath) and options.is_update_policy:
        print('remove old model')
        os.remove(model_fpath)

    timesteps = 122

    skill_list = ['open_drawer', 'close_drawer']
    exp = CALVINExp(skill_list=skill_list)
    q0 = exp.skill_ds[0]

    # skill_dim = 6
    skill_dim = sum([exp.datasets[i].X.numpy().shape[-1] for i in range(len(exp.skill_list))])

    Xt_d, dXt_d, desired_ = exp.get_taskspace_training_data()
    timestamps = np.linspace(0, 1, timesteps - 1)
    feat = timestamps[:, np.newaxis]
    dataset = SkillDataset(feat, Xt_d, dXt_d, desired_)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    # Load skills from skill library
    # Get the policy type and parameters
    policy_config = {'dim': skill_dim,
                     'fdim': 1,
                     'skill': batch_skill}
    policy_type = getattr(policy_classes, options.policy_name)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--policy", dest="policy_name", type="string",
                      default="SkillDiagonalWeightedPolicy",
                      # options='SkillDiagonalWeightedPolicy, SkillFullyWeightedPolicy, '
                      help="Set policy to train or to test. "
                           "Note: the stored policy file should coincide with the policy"
                           "type, otherwise, use --update_policy)")
    parser.add_option("-d", "--data_file", dest="data_file", type="string",
                      default="examples/GripperExperiment/demos/pickplace_demo.npz",
                      help="Set the data filename (to store or update)")
    parser.add_option("-p", "--policy_file", dest="policy_file", type="string",
                      default="examples/GripperExperiment/trained_policies/policy",
                      help="Set the model filename (to store or update)")
    parser.add_option("-u", "--update_policy", action="store_true", dest="is_update_policy", default=True,
                      help="Store (or update) the policy in current folder")
    parser.add_option("--show_demo_test", action="store_true", dest="is_show_demo_test", default=True,
                      help="Show the learned policy")
    parser.add_option("--generalized", action="store_true", dest="is_generalized", default=False,
                      help="Show the learned policy")
    (options, args) = parser.parse_args(sys.argv)

    CALVIN_experiment(options)