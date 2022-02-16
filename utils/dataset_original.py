from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from itertools import chain, product
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import copy 
import random
import torchvision
from torchvision import transforms
import torch.nn.functional as F
# NOTE: 363 poses total

# NOTE: 10% of keyframes are eval


class BasicDataset(Dataset):
    def __init__(self, pd, scale=1, mask_suffix='', mode='show_room'):
        """
        i need to split this above by keyframes
        LATER, KFOLD KEYFRAMES AND RANDOMLY SELECT KEYFRAME AND RANDOMLY SELECT POSE WITHIN FRAME
        """
        self.data_mode = 'train'
        self.mode = mode
        self.parent_dir = pd
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.img_files_R = [x[0] for x in os.walk(self.parent_dir) if len(x[0].split('/')) == 6]

        logging.info(f'Creating dataset with {len(self.img_files_R)} examples')

    def __len__(self):
        return len(self.img_files_R)

    #@classmethod
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

    #@classmethod
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

    def __getitem__(self, i):
        """
        i will get the random keyframe, 
        then i need to find len and randomise the pose selection
        """
        random_pose_idx = random.randint(0, len(os.listdir(self.img_files_R[i]+'/ambient_light/'))-1)
        pose = os.listdir(self.img_files_R[i]+'/ambient_light/')[random_pose_idx][0] # list all unique poses in the dir and then random the list
        img_dir_R = self.img_files_R[i] + '/ambient_light/' + str(pose) + '_R.png'
        img_file_R = glob(img_dir_R)
        img_file_L = glob(img_dir_R.replace('_R', '_L'))

        mask_file_R = []
        mask_file_L = []
        for xx in range(9):
            mask_dir_R = img_dir_R.replace('ambient_light','pattern_'+str(xx))
            mask_file_R.append(glob(mask_dir_R.replace('_R', '_B_r')))
            mask_file_L.append(glob(mask_dir_R.replace('_R', '_B_l')))
        mask_file_R = np.array(mask_file_R)
        mask_file_L = np.array(mask_file_L)

        assert len(img_file_R) == 1, \
            f'no file found'


        img_R = self.preprocess_input(Image.open(img_file_R[0]), self.scale)#.convert('L'), self.scale)
        img_L = self.preprocess_input(Image.open(img_file_L[0]), self.scale)#.convert('L'), self.scale)

        rr1=np.array(Image.open(img_dir_R.replace('ambient_light', 'three_phase').replace('_R', '_tp1_r'))) - np.array(Image.open(img_dir_R.replace('ambient_light', 'three_phase').replace('_R', '_tp2_r')))
        rr2=np.array(Image.open(img_dir_R.replace('ambient_light', 'three_phase').replace('_R', '_tp1_r'))) - np.array(Image.open(img_dir_R.replace('ambient_light', 'three_phase').replace('_R', '_tp3_r')))
        rr3=np.array(Image.open(img_dir_R.replace('ambient_light', 'three_phase').replace('_R', '_tp2_r'))) - np.array(Image.open(img_dir_R.replace('ambient_light', 'three_phase').replace('_R', '_tp3_r')))
        modulation_right = (2 * np.sqrt(2) / 3) * np.sqrt(
                (rr1) ** 2 +
                (rr2) ** 2 +
                (rr3) ** 2)

        ll1=np.array(Image.open(img_dir_R.replace('ambient_light', 'three_phase').replace('_R', '_tp1_l'))) - np.array(Image.open(img_dir_R.replace('ambient_light', 'three_phase').replace('_R', '_tp2_l')))
        ll2=np.array(Image.open(img_dir_R.replace('ambient_light', 'three_phase').replace('_R', '_tp1_l'))) - np.array(Image.open(img_dir_R.replace('ambient_light', 'three_phase').replace('_R', '_tp3_l')))
        ll3=np.array(Image.open(img_dir_R.replace('ambient_light', 'three_phase').replace('_R', '_tp2_l'))) - np.array(Image.open(img_dir_R.replace('ambient_light', 'three_phase').replace('_R', '_tp3_l')))
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

        focal_len = 911.7019228756361
        baseline = 5.563167785169519
        num = focal_len*baseline

        depth_r = np.load(img_dir_R.replace('ambient_light', 'depth').replace('_R.png', '_depth_R.npy'))
        depth_l = np.load(img_dir_R.replace('ambient_light', 'depth').replace('_R.png', '_depth_L.npy'))

        depth_r = Image.fromarray(depth_r)
        depth_l = Image.fromarray(depth_l)
        w, h = depth_r.size
        newW, newH = int(self.scale * w), int(self.scale * h) 
        depth_r = np.expand_dims(np.array(depth_r.resize((newW, newH))),0)
        depth_l = np.expand_dims(np.array(depth_l.resize((newW, newH))),0)
        depth_r_mask = (depth_r > 1e-9)*1.
        depth_l_mask = (depth_l > 1e-9)*1.
        depth_r_mask_tiled = np.tile(depth_r_mask,(9, 1, 1))
        depth_l_mask_tiled = np.tile(depth_l_mask,(9, 1, 1))
        depth_mask = np.concatenate((depth_r_mask_tiled, depth_l_mask_tiled),0)

        disparity_l_to_r = np.maximum(np.minimum(num/(depth_l+1e-9),newW*1.),0.)/(newW*1.)
        disparity_r_to_l = (np.maximum(np.minimum(num/(depth_r+1e-9),newW*1.),0.)*-1.)/(newW*1.)
        disparity = np.concatenate((disparity_r_to_l,disparity_l_to_r),0)



        img = np.concatenate((img_R,img_L),0) # I will be stacking the left and right
        mask = np.round(np.concatenate((mask_R,mask_L),0))




        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'disparity': torch.from_numpy(disparity).type(torch.FloatTensor),
            'depth_mask': torch.from_numpy(depth_mask).type(torch.FloatTensor)
        }

