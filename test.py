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


train_loader = datasets.fetch_dataloader(args)
train_loader.__getitem__(0)