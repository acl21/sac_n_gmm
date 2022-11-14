import os
import inspect

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
logging.basicConfig(filename=f'./examples/CALVINExperiment/logs/train_ds_{time_now}.log', encoding='utf-8', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d.%m.%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)

def train_skills_ds(options):
    # Get skill list
    f = open(options.skill_list_file, "r")
    skill_set = f.read()
    skill_set = skill_set.split("\n")
    logger.info(f'Found {len(skill_set)} skills in the list')
    logger.info(f'Training DS with {options.state_type} as the input')
    for idx, skill in enumerate(skill_set):
        # Create skill specific output directory
        skill_output_dir = os.path.join(options.ds_output_dir, options.state_type, skill)
        os.makedirs(skill_output_dir, exist_ok=True)
        # Get train and validation datasets
        train_dataset = CALVINDynSysDataset(skill=skill, state_type=options.state_type, demos_dir=options.demos_dir, goal_centered=True)
        val_dataset = CALVINDynSysDataset(skill=skill, state_type=options.state_type, train=False, demos_dir=options.demos_dir, goal_centered=True)
        logger.info(f'Skill {idx}: {skill}, Train Data: {train_dataset.X.size()}, Val. Data: {val_dataset.X.size()}')
        # Create models to train
        dim = train_dataset.X.shape[-1]
        clf_model = WSAQF(dim=dim, n_qfcn=1)
        reg_model = SimpleNN(in_dim=dim, out_dim=dim, n_layers=(20, 20))
        clfds = CLFDS(clf_model, reg_model, rho_0=0.1, kappa_0=0.0001)
        # Training
        if options.train_clf:
            clfds.train_clf(train_dataset, val_dataset, lr=options.lr, max_epochs=options.max_epochs,\
                batch_size=options.batch_size, fname=os.path.join(skill_output_dir, 'clf'))
        else:
            clfds.train_ds(train_dataset, val_dataset, lr=options.lr, max_epochs=options.max_epochs,\
                batch_size=options.batch_size, fname=os.path.join(skill_output_dir, 'ds'))
    logger.info(f'Training complete. Trained DS models are saved in {os.path.join(options.ds_output_dir, options.state_type)} directory')

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
    parser.add_option("--clf", dest="train_clf", action='store_true',
                      help="Should CLF model be trained?")
    parser.add_option("--reg", dest="train_clf", action='store_false', default=False,
                      help="Should REG model be trained?")
    parser.add_option("--max-epochs", dest="max_epochs", type='int',
                      default=1000, help="Number of epochs to train each DS")
    parser.add_option("--batch-size", dest="batch_size", type='int',
                      default=32, help="Batch Size")
    parser.add_option("--lr", dest="lr", type='float',
                      default=1e-3, help="Learning Rate")
    parser.add_option("--ds-output-dir", dest="ds_output_dir", type='string',
                      default='./examples/CALVINExperiment/skills_ds/', help="Path to the DS output directory")

    (options, args) = parser.parse_args(sys.argv)
    train_skills_ds(options)