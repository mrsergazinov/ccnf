# -*- coding: utf-8 -*-
# ---------------------

import os

PYTHONPATH = '..:.'
if os.environ.get('PYTHONPATH', default=None) is None:
    os.environ['PYTHONPATH'] = PYTHONPATH
else:
    os.environ['PYTHONPATH'] += (':' + PYTHONPATH)

import yaml
import socket
import random
import torch
import numpy as np
from path import Path
from typing import Optional
import termcolor
from datetime import datetime


def set_seed(seed=None):
    # type: (Optional[int]) -> int
    """
    set the random seed using the required value (`seed`)
    or a random value if `seed` is `None`
    :return: the newly set seed
    """
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


class Config(object):
    HOSTNAME = socket.gethostname()
    LOG_PATH = Path('./logs/')

    def __init__(self, conf_file_path=None, seed=None, exp_name=None, device=None, log=True):
        # type: (str, int, str, bool) -> None
        """
        :param conf_file_path: optional path of the configuration file
        :param seed: desired seed for the RNG; if `None`, it will be chosen randomly
        :param exp_name: name of the experiment
        :param: what device to use for training
        :param log: `True` if you want to log each step; `False` otherwise
        """
        self.exp_name = exp_name
        self.log_each_step = log

        # print project name and host name
        self.project_name = Path(__file__).parent.parent.basename()
        m_str = f'┃ {self.project_name}@{Config.HOSTNAME} ┃'
        u_str = '┏' + '━' * (len(m_str) - 2) + '┓'
        b_str = '┗' + '━' * (len(m_str) - 2) + '┛'
        print(u_str + '\n' + m_str + '\n' + b_str)

        # define output paths
        self.project_log_path = Path('./log')

        # set random seed
        self.seed = set_seed(seed)  # type: int

        self.keys_to_hide = list(self.__dict__.keys()) + ['keys_to_hide']

        # if the configuration file is not specified
        # try to load a configuration file based on the experiment name
        tmp = Path(__file__).parent / (self.exp_name + '.yaml')
        if conf_file_path is None and tmp.exists():
            conf_file_path = tmp

        # read the YAML configuation file
        if conf_file_path is None:
            y = {}
        else:
            conf_file = open(conf_file_path, 'r')
            y = yaml.load(conf_file, Loader=yaml.Loader)

        # read configuration parameters from YAML file
        # or set their default value
        self.ds_name = y.get('ds_name', "electricity")  # type: str
        self.exp_id = y.get('exp_id', str(self.seed))  # type: str
        self.total_time_steps = y.get('total_time_steps', 1)  # type: int
        self.num_encoder_steps = y.get('num_encoder_steps', 1) # type: int
        self.num_decoder_steps = self.total_time_steps - self.num_encoder_steps # type: int
        self.num_flows = y.get('num_flows', 1)  # type: int
        self.hidden = y.get('hidden', 128)  # type: int
        self.max_grad_norm = y.get('max_gradient_norm', 1.0)  # type: float
        self.lr = y.get('lr', 0.0001)  # type: float
        self.num_epochs = y.get('num_epochs', 100)  # type: int
        self.early_stopping = y.get('early_stopping', 10)  # type: int
        self.n_workers = y.get('n_workers', 1)  # type: int
        self.batch_size = y.get('batch_size', 64)  # type: int
        self.all_params = y  # type: dict
        self.exp_log_path = self.project_log_path / exp_name / datetime.now().strftime(
            "%m.%d.%Y_%H.%M.%S")

        self.device = device

    def write_to_file(self, out_file_path):
        # type: (str) -> None
        """
        Writes configuration parameters to `out_file_path`
        :param out_file_path: path of the output file
        """
        import re

        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        text = ansi_escape.sub('', str(self))
        with open(out_file_path, 'w') as out_file:
            print(text, file=out_file)

    def __str__(self):
        # type: () -> str
        out_str = ''
        for key in self.__dict__:
            if key in self.keys_to_hide:
                continue
            value = self.__dict__[key]
            out_str += str(key.upper()) + ': ' + str(value) + '\n'
        return out_str[:-1]

    def no_color_str(self):
        # type: () -> str
        out_str = ''
        for key in self.__dict__:
            value = self.__dict__[key]
            if type(value) is Path or type(value) is str:
                value = value.replace(Config.LOG_PATH, '$LOG_PATH')
            out_str += f'{key.upper()}: {value}\n'
        return out_str[:-1]


def show_default_params():
    """
    Print default configuration parameters
    """
    cnf = Config(exp_name='default')
    print(f'\nDefault configuration parameters: \n{cnf}')


if __name__ == '__main__':
    show_default_params()
