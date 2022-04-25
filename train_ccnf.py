import argparse
from model.solver import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='electricity', help='[electricity, ...]')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batches_per_epoch', type=int, default=2000, help='number of epochs')
parser.add_argument('--early_stopping', type=int, default=None, help='epochs to early stopping')
parser.add_argument('--length', type=int, default=24*7, help='history length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction length')
parser.add_argument('--hidden', type=int, default=1024, help='hidden dimension')
parser.add_argument('--gpu', type=int, default=1, help='use cuda')
parser.add_argument('--gpu_id', type=int, default=1, help='gpu id to use')
parser.add_argument('--save_path', type=str, default='./checkpoints', help='directory to save models')

args = parser.parse_args()
args.gpu = True if args.gpu==1 else False

if __name__ == "__main__":
    solver = CCNF(args.dataset, args.batch_size, args.length, args.pred_len, args.hidden, args.gpu, args.gpu_id, args.save_path)
    solver.fit(args.epochs, args.batches_per_epoch, args.early_stopping)
