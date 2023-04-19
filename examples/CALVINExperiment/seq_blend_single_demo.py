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
from examples.CALVINExperiment.CALVINSkill import CALVINSkillComplex
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

    # Define the loss weights to define the importance of the skills in the loss function
    loss_weights = np.array([1.0, 1.0])

    timesteps = 122

    skill_names = ['open_drawer', 'close_drawer']
    exp = CALVINExp(skill_names=skill_names)
    q0 = exp.skill_list[0]

    # skill_dim = 6
    skill_dim = sum([exp.skill_list[i].dataset.X.numpy().shape[-1] for i in range(len(exp.skill_list))])

    Xt_d, dXt_d, desired_ = exp.get_taskspace_training_data()
    batch_skill = CALVINSkillComplex(skill_dim, skill_dim, exp.skill_list, skill_cluster_idx=None)

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
    constraints = {'eqn': None, 'ineqn': None}

    # Create the policy
    policy = policy_type(**policy_config, **constraints)
    policy.to(device)

    if not os.path.exists(model_fpath):
        # Train the policy
        # If the policy has a full weight matrix, we first train a policy with diagonal weight matrix
        if policy_type == policy_classes.SkillFullyWeightedPolicy:
            diag_policy = policy_classes.SkillDiagonalWeightedPolicy(**policy_config, **constraints)
            diag_policy.to(device)
            diag_policy = policy_classes.train_policy(diag_policy, dataloader, loss_weights=loss_weights,
                                                      model_path=model_fpath,
                                                      learning_rate=learning_rate,
                                                      max_epochs=MAX_EPOCH,
                                                      consider_spec_train=False)
            # The diagonal part of the full weight matrix is then initialized with the pretrained diagonal policy
            # The full weight matrix is then trained starting at this initial point
            policy.QDiagNet = diag_policy.Qnet

        policy = policy_classes.train_policy(policy, dataloader, loss_weights=loss_weights,
                                             model_path=model_fpath,
                                             learning_rate=learning_rate,
                                             max_epochs=MAX_EPOCH,
                                             consider_spec_train=False)
    else:
        # If the policy already exists, load it
        policy.load_policy(model_fpath)
        policy.skill = batch_skill
        policy.to(device)


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