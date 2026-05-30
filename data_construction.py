import argparse
from processor import Preprocess
import os
import sys

run_dir = os.path.dirname(sys.argv[0])
if run_dir == '':
    run_dir = '.'
os.chdir(run_dir)

parser = argparse.ArgumentParser(description="AE-based 3D-CNN pretrain")
parser.add_argument('--dataset', type=str, default='IP')
parser.add_argument('--encoded_dim', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--windowsize', type=int, default=27)
parser.add_argument('--Patch_channel', type=int, default=15)
parser.add_argument('--num_clients', type=int, default=5, help="Number of clients for splitting dataset")
args = parser.parse_args()

if args.dataset == 'IP':
    args.Patch_channel = 30

XPath = f'DataArray/{args.dataset}_X.npy'
Preprocess(XPath, args.dataset, args.windowsize, Patch_channel=args.Patch_channel, num_splits=args.num_clients)