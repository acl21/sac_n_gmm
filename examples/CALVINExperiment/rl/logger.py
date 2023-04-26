import os
import csv
import torch
import wandb
import shutil
import numpy as np
from termcolor import colored
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

COMMON_TRAIN_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('env_step', 'EnvS', 'int'),
    ('episode_reward', 'R', 'float'),
    ('duration', 'D', 'time') 
]

COMMON_EVAL_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('episode_reward', 'R', 'float') 
]


AGENT_TRAIN_FORMAT = {
    'seqblend-sac': [
        ('batch_reward', 'BR', 'float'),
        ('actor_loss', 'ALOSS', 'float'),
        ('critic_loss', 'CLOSS', 'float'),
        ('alpha_loss', 'TLOSS', 'float'),
        ('alpha_value', 'TVAL', 'float'),
        ('actor_entropy', 'AENT', 'float')
    ]
}

class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)

class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._csv_file_name = self._prepare_file(file_name, 'csv')
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = open(self._csv_file_name, 'w')
        self._csv_writer = None

    def _prepare_file(self, prefix, suffix):
        file_name = f'{prefix}.{suffix}'
        if os.path.exists(file_name):
            os.remove(file_name)
        return file_name

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            self._csv_writer.writeheader()
        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            return f'{key}: {value:04.1f} s'
        else:
            raise f'invalid format type: {ty}'

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = [f'| {prefix: <14}']
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    def dump(self, step, prefix, save=True):
        if len(self._meters) == 0:
            return
        if save:
            data = self._prime_meters()
            data['step'] = step
            self._dump_to_csv(data)
            self._dump_to_console(data, prefix)
        self._meters.clear()

class Logger(object):
    def __init__(self,
                 log_dir,
                 save_wb=False,
                 cfg=None,
                 log_frequency=10000,
                 agent='seqblend-sac'):
        self._log_dir = log_dir
        self._log_frequency = log_frequency

        self.save_wb = save_wb
        if self.save_wb:
            config = {
            "num_train_steps": cfg.num_train_steps,
            "num_seed_steps": cfg.num_seed_steps,
            "eval_frequency": cfg.eval_frequency,
            "num_eval_episodes": cfg.num_eval_episodes,
            "accumulate_steps": cfg.accumulate_steps,
            "env_max_episode_steps": cfg.env_max_episode_steps,
            }
            wandb.init(project=agent, entity='in-ac', config=config)


        # each agent has specific output format for training
        assert agent in AGENT_TRAIN_FORMAT
        train_format = COMMON_TRAIN_FORMAT + AGENT_TRAIN_FORMAT[agent]
        self._train_mg = MetersGroup(os.path.join(log_dir, 'train'),
                                     formating=train_format)
        self._eval_mg = MetersGroup(os.path.join(log_dir, 'eval'),
                                    formating=COMMON_EVAL_FORMAT)

    def log(self, key, value):
        if 'video' in key:
            self.log_video(key, value)
            return
        if type(value) == torch.Tensor:
            value = value.item()
        if self.save_wb:
            wb_key = key.replace('/', '_')
            wandb.log({wb_key: value})
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value, 1)

    def log_video(self, key, filepath):
        if self.save_wb:
            wb_key = key.split('/')[-1]
            wandb.log(
            {wb_key: wandb.Video(filepath, fps=15, format="gif")})

    def log_params(self, agent, fname=None, actor=False, critic=False):
        self.weights_dir = os.path.join(self._log_dir, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)
        if fname is None:
            fname = ''
        if actor:
            torch.save(agent.actor.trunk.state_dict(), os.path.join(self.weights_dir, f"actor_{fname}.pth"))
            torch.save(agent.critic.Q1.state_dict(), os.path.join(self.weights_dir, f"critic_q1_{fname}.pth"))
        if critic:
            torch.save(agent.critic.Q2.state_dict(), os.path.join(self.weights_dir, f"critic_q2_{fname}.pth"))

    def dump(self, step, save=True, ty=None):
        if ty is None:
            self._train_mg.dump(step, 'train', save)
            self._eval_mg.dump(step, 'eval', save)
        elif ty == 'eval':
            self._eval_mg.dump(step, 'eval', save)
        elif ty == 'train':
            self._train_mg.dump(step, 'train', save)
        else:
            raise f'invalid log type: {ty}'