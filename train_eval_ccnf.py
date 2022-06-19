import os
import argparse
from config.setup import Config
from model.solver import *

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--conf_file_path', type=str, default=None)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()
args.gpu = 'cuda:{}'.format(args.gpu) if args.gpu >= 0 else 'cpu'

if __name__ == "__main__":
    conf = Config(conf_file_path=args.conf_file_path, seed=args.seed, device=args.gpu, exp_name=args.exp_name)
    conf.write_to_file(os.path.join(conf.exp_log_path, 'conf.txt'))
    print(f'\n{conf}')

    solver = CCNF(conf)
    solver.fit()
    solver.evaluate()
