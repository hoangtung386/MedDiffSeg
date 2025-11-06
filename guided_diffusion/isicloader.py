import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate

class ISICDataset(Dataset):
    def __init__(self, args, data_path , transform = None, mode = 'Training',plane = False):


        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode

        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        if self.transform:
            state = torch.get_rng_state()
            img_2d = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)
        else:
            # Fallback to ToTensor if no transform is provided
            img_2d = F.to_tensor(img)
            mask = F.to_tensor(mask)

        # Create a grayscale version for the 2.5D path, as the 2.5D encoder expects a single channel
        img_gray = F.rgb_to_grayscale(img_2d)

        # Create the pseudo 2.5D volume by stacking the grayscale image.
        # The model's 2.5D encoder expects (B, C, D, H, W) where C=1.
        # We create a (1, 3, H, W) tensor, where D=3.
        img_2_5d = img_gray.unsqueeze(1).repeat(1, 3, 1, 1)

        return ((img_2d, img_2_5d), mask, name)