from __future__ import print_function, division

import argparse
import logging
import numpy as np
# from pathlib import Path
# from tqdm import tqdm

# from torch.utils.tensorboard import SummaryWriter
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from core.raft_stereo import RAFTStereo

# from evaluate_stereo import *
# import core.stereo_datasets as datasets
import core.sl_datasets as datasets
# from utils.dataset import BasicDataset
# from torch.utils.data import DataLoader, random_split

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='raft-stereo', help="name your experiment")
parser.add_argument('--restore_ckpt', help="restore checkpoint")
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

# Training parameters
parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
parser.add_argument('--image_size', type=int, nargs='+', default=[320, 720], help="size of the random image crops used during training.")
parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")
parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

# Validation parameters
parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

# Architecure choices
parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")

# Data augmentation
parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
args = parser.parse_args()

# torch.manual_seed(1234)
# np.random.seed(1234)

train_loader = datasets.fetch_dataloader(args)
# train_loader.__getitem__(0)
train_loader[0]