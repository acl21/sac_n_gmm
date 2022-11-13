import os
import inspect
import csv

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
main_dir = current_dir + '/../../'
os.sys.path.insert(0, '../' + current_dir)
os.sys.path.insert(0, main_dir)

import sys
from optparse import OptionParser

from SkillsSequencing.skills.mps.dynsys.CLFDS import CLFDS
from SkillsSequencing.skills.mps.dynsys.WSAQF import WSAQF
from SkillsSequencing.skills.mps.dynsys.FNN import SimpleNN
from SkillsSequencing.skills.mps.dynsys.CALVIN_DS import CALVINDynSysDataset

import datetime
time_now = datetime.datetime.now().strftime("%m%d-%H%M%S")
import logging
logging.basicConfig(filename=f'./examples/CALVINExperiment/logs/evaluate_ds_{time_now}.log', encoding='utf-8', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d.%m.%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)

def evaluate_skills_ds(options):
    # Get skill list
    f = open(options.skill_list_file, "r")
    skill_set = f.read()
    skill_set = skill_set.split("\n")
    logger.info(f'Found {len(skill_set)} skills in the list')
    logger.info(f'Evaluating skills DS with {options.state_type} as the input')
    skill_accs = {}
    for idx, skill in enumerate(skill_set):
        # Get skill file output directory
        skill_output_dir = os.path.join(options.ds_model_dir, options.state_type, skill)
        clf_file = os.path.join(skill_output_dir, 'clf')
        reg_file = os.path.join(skill_output_dir, 'ds')
        # Get train and validation datasets
        val_dataset = CALVINDynSysDataset(skill=skill, state_type=options.state_type, train=False, demos_dir=options.demos_dir)
        logger.info(f'Skill {idx}: {skill}, Test/Val Data: {val_dataset.X.size()}')
        # Create and load models to evaluate
        dim = val_dataset.X.shape[-1]
        clf_model = WSAQF(dim=dim, n_qfcn=1)
        reg_model = SimpleNN(in_dim=dim, out_dim=dim, n_layers=(20, 20))
        clfds = CLFDS(clf_model, reg_model, rho_0=0.1, kappa_0=0.0001)
        clfds.load_models(clf_file=clf_file, reg_file=reg_file)
        # Evaluating
        acc = clfds.evaluate_ds(val_dataset, error_margin=options.error_margin,\
                T=options.timesteps, speed=options.speed)
        skill_accs[skill] = str(acc)

    # Write accuracies to a file
    with open(os.path.join('./examples/CALVINExperiment/logs/', f'skill_ds_acc_{options.state_type}_er{options.error_margin}_sp{options.speed}.txt'), 'w') as f:
        writer = csv.writer(f)
        for row in skill_accs.items():
            print(row)
            writer.writerow(row)
    logger.info(f'Evaluation complete. DS model accuracies are saved at {os.path.join("./examples/CALVINExperiment/logs/", f"skill_ds_acc_{options.state_type}_er{options.error_margin}_sp{options.speed}.txt")} directory')

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--skill-list-file", dest="skill_list_file", type='string',
                      default='/work/dlclarge1/lagandua-refine-skills/calvin_demos/skillnames.txt',
                      help='Path to a text file with all skill names')
    parser.add_option("--demos-dir", dest="demos_dir", type='string',
                      default='/work/dlclarge1/lagandua-refine-skills/calvin_demos/',
                      help="Path to the saved demos directory")
    parser.add_option("--state-type", dest="state_type", type='string',
                      default='joint', help="What should be consired as input to the DS model? \
                        Options: joint -> joint angles, pos - EE 3D position, ori - EE 3D orientation \
                           pos_ori -> EE position + orientation, grip -> gripper width")
    parser.add_option("--error-margin", dest="error_margin", type='float',
                      default=0.01, help="Error margin (norm ball) to consider the EE to have reached the goal")
    parser.add_option("--timesteps", dest="timesteps", type='int',
                      default=1000, help="Max timesteps to evaluate each trajectory")
    parser.add_option("--speed", dest="speed", type='float',
                      default=0.01, help="EE Speed (X + speed * d_X)")
    parser.add_option("--ds-model-dir", dest="ds_model_dir", type='string',
                      default='./examples/CALVINExperiment/skills_ds/', help="Path to the DS output directory")

    (options, args) = parser.parse_args(sys.argv)
    evaluate_skills_ds(options)