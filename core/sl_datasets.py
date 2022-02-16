# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp
from PIL import Image
import torchvision
from torchvision import transforms

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor

class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader        

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []
        self.data_mode = 'train'

    def preprocess(self, pil_img, scale): #(cls, pil_img, scale): 

        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))


        img_nd = np.array(pil_img)


        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans

    def preprocess_input(self, pil_img, scale): #(cls, pil_img, scale): 
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd#.transpose((2, 0, 1))


        if self.data_mode == 'train':
                tx = torchvision.transforms.Compose([
                        torchvision.transforms.ToPILImage(),
                        #torchvision.transforms.RandomRotation((-3,3)),
                        torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                        #torchvision.transforms.
                ])
        elif self.data_mode == 'valid':
                tx = torchvision.transforms.Compose([
                        torchvision.transforms.ToPILImage(),
                ])
        img_trans = np.array(tx(img_trans))
        img_trans = img_nd.transpose((2, 0, 1))

        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def structlight_reader(self, scene_num, pose_num):
        img_file_R = self.data_path + '{}/ambient_light/{}_R.png'.format(scene_num, pose_num)
        img_file_L = self.data_path + '{}/ambient_light/{}_L.png'.format(scene_num, pose_num)

        mask_file_R = []
        mask_file_L = []
        for xx in range(9):
            mask_dir_R = img_file_R.replace('ambient_light','pattern_'+str(xx))
            mask_file_R.append(glob(mask_dir_R.replace('_R', '_B_r')))
            mask_file_L.append(glob(mask_dir_R.replace('_R', '_B_l')))
        mask_file_R = np.array(mask_file_R)
        mask_file_L = np.array(mask_file_L)

        img_R = self.preprocess_input(Image.open(img_file_R), self.scale)#.convert('L'), self.scale)
        img_L = self.preprocess_input(Image.open(img_file_L), self.scale)#.convert('L'), self.scale)

        rr1=np.array(Image.open(img_file_R.replace('ambient_light', 'three_phase').replace('_R', '_tp1_r'))) - np.array(Image.open(img_file_R.replace('ambient_light', 'three_phase').replace('_R', '_tp2_r')))
        rr2=np.array(Image.open(img_file_R.replace('ambient_light', 'three_phase').replace('_R', '_tp1_r'))) - np.array(Image.open(img_file_R.replace('ambient_light', 'three_phase').replace('_R', '_tp3_r')))
        rr3=np.array(Image.open(img_file_R.replace('ambient_light', 'three_phase').replace('_R', '_tp2_r'))) - np.array(Image.open(img_file_R.replace('ambient_light', 'three_phase').replace('_R', '_tp3_r')))
        modulation_right = (2 * np.sqrt(2) / 3) * np.sqrt(
                (rr1) ** 2 +
                (rr2) ** 2 +
                (rr3) ** 2)

        ll1=np.array(Image.open(img_file_R.replace('ambient_light', 'three_phase').replace('_R', '_tp1_l'))) - np.array(Image.open(img_file_R.replace('ambient_light', 'three_phase').replace('_R', '_tp2_l')))
        ll2=np.array(Image.open(img_file_R.replace('ambient_light', 'three_phase').replace('_R', '_tp1_l'))) - np.array(Image.open(img_file_R.replace('ambient_light', 'three_phase').replace('_R', '_tp3_l')))
        ll3=np.array(Image.open(img_file_R.replace('ambient_light', 'three_phase').replace('_R', '_tp2_l'))) - np.array(Image.open(img_file_R.replace('ambient_light', 'three_phase').replace('_R', '_tp3_l')))
        modulation_left = (2 * np.sqrt(2) / 3) * np.sqrt(
                (ll1) ** 2 +
                (ll2) ** 2 +
                (ll3) ** 2)
        if self.data_mode == 'train':
                random_uncertainty = abs(10 + 9 * np.random.randn(1))[0]
                mask_uncer_l = (modulation_left > random_uncertainty) * 1.
                mask_uncer_r = (modulation_right > random_uncertainty) * 1.
        elif self.data_mode == 'valid':
                mask_uncer_l = (modulation_left > 5) * 1.
                mask_uncer_r = (modulation_right > 5) * 1.

        mask_R = []
        mask_L = []
        for xx in range(9):
            right_image = Image.fromarray(np.uint8(np.array(Image.open(mask_file_R[xx][0]).convert('L')) * mask_uncer_r))
            mask_R.append(np.expand_dims(self.preprocess(right_image, self.scale)[0],0))
            left_image = Image.fromarray(np.uint8(np.array(Image.open(mask_file_L[xx][0]).convert('L')) * mask_uncer_l))
            mask_L.append(np.expand_dims(self.preprocess(left_image, self.scale)[0],0))
        mask_R = np.squeeze(np.array(mask_R))
        mask_L = np.squeeze(np.array(mask_L))
        mask = np.round(np.concatenate((mask_R,mask_L),0))

        return img_L, img_R, mask
    
    def __getitem__(self, index):
        # if not self.init_seed:
        #     worker_info = torch.utils.data.get_worker_info()
        #     if worker_info is not None:
        #         torch.manual_seed(worker_info.id)
        #         np.random.seed(worker_info.id)
        #         random.seed(worker_info.id)
        #         self.init_seed = True

        result = self.imgR_list[index].split('/')
        scene_num = result[-3]
        pose_num = result[-1][:-6]

        img1, img2, disp = self.structlight_reader(scene_num, pose_num)

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        disp = np.array(disp).astype(np.float32)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        return img1, img2, disp

    ### From original code to duplicate the class attribute
    # def __mul__(self, v):
    #     copy_of_self = copy.deepcopy(self)
    #     copy_of_self.flow_list = v * copy_of_self.flow_list
    #     copy_of_self.image_list = v * copy_of_self.image_list
    #     copy_of_self.disparity_list = v * copy_of_self.disparity_list
    #     copy_of_self.extra_info = v * copy_of_self.extra_info
    #     return copy_of_self
        
    def __len__(self):
        return len(self.image_list)


class StructLight(StereoDataset):
    def __init__(self, aug_params=None, data_path='/home/terryxu/Downloads/hx/', split='training'):
        super(StructLight, self).__init__(aug_params, sparse=True)

        with open(os.path.join('SL', 'img_r_list_full.txt'), 'r') as f:
            lines = f.readlines()
        self.imgR_list = lines
        self.data_path = data_path
        scale = 1/2
        self.scale = scale
  
def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    new_dataset = StructLight(aug_params)
    logging.info(f"Adding {len(new_dataset)} samples from Alister's Dataset")
    train_dataset = new_dataset
    return train_dataset

    # train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
    #     pin_memory=True, shuffle=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=True)

    # logging.info('Training with %d image pairs' % len(train_dataset))
    # return train_loader

