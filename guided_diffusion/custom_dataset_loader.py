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
from glob import glob
from sklearn.model_selection import train_test_split
import nibabel


class CustomDataset(Dataset):
    def __init__(self, args, data_path, transform=None, mode="Training", plane=False):

        print("loading data from the directory :", data_path)
        path = data_path
        images = sorted(glob(os.path.join(path, "images/*.png")))
        masks = sorted(glob(os.path.join(path, "masks/*.png")))

        self.name_list = images
        self.label_list = masks
        self.data_path = path
        self.mode = mode

        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(name)

        mask_name = self.label_list[index]
        msk_path = os.path.join(mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(msk_path).convert("L")

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        return (img, mask, name)
        # if self.mode == 'Training':
        #     return (img, mask, name)
        # else:
        #     return (img, mask, name)


class CustomDataset3D(torch.utils.data.Dataset):
    def __init__(self, data_path, transform, num_slices_2_5d=3):
        super().__init__()
        
        print("loading data from the directory :", data_path)
        path = data_path
        images = sorted(glob(os.path.join(path, "images/*.nii.gz")))
        masks = sorted(glob(os.path.join(path, "masks/*.nii.gz")))

        assert len(images) == len(masks), "Number of images and masks must be the same"
        
        self.valid_cases = [(img_path, seg_path) for img_path, seg_path in zip(images, masks)]

        self.all_slices = []
        self.case_num_slices = []
        for case_idx, (img_path, seg_path) in enumerate(self.valid_cases):
            img = nibabel.load(img_path)
            seg = nibabel.load(seg_path)
            assert (
                img.shape == seg.shape
            ), f"Image and segmentation shape mismatch: {img.shape} vs {seg.shape}, Files: {img_path}, {seg_path}"
            num_slices = img.shape[-1]
            self.case_num_slices.append(num_slices)
            self.all_slices.extend(
                [(case_idx, slice_idx) for slice_idx in range(num_slices)]
            )
            
        self.data_path = path
        self.num_slices_2_5d = num_slices_2_5d
        self.transform = transform

    def __len__(self):
        return len(self.all_slices)

    def __getitem__(self, x):
        case_idx, slice_idx = self.all_slices[x]
        img_path, seg_path = self.valid_cases[case_idx]

        nib_img = nibabel.load(img_path)
        nib_seg = nibabel.load(seg_path)

        img_data = nib_img.get_fdata()
        seg_data = nib_seg.get_fdata()

        # 2D data (center slice)
        image_2d = torch.tensor(img_data, dtype=torch.float32)[:, :, slice_idx].unsqueeze(0).unsqueeze(0)
        label = torch.tensor(seg_data, dtype=torch.float32)[:, :, slice_idx].unsqueeze(0).unsqueeze(0)
        label = torch.where(label > 0, 1, 0).float()

        # 2.5D data
        half_slices = self.num_slices_2_5d // 2
        start_slice = slice_idx - half_slices
        end_slice = slice_idx + half_slices + 1
        
        slices_2_5d = []
        num_slices_total = self.case_num_slices[case_idx]
        for i in range(start_slice, end_slice):
            # Pad with nearest slice at boundaries
            slice_to_get = np.clip(i, 0, num_slices_total - 1)
            slices_2_5d.append(torch.tensor(img_data, dtype=torch.float32)[:, :, slice_to_get])
        
        image_2_5d = torch.stack(slices_2_5d, dim=0).unsqueeze(0) # Shape: (1, D, H, W)

        if self.transform:
            state = torch.get_rng_state()
            image_2d = self.transform(image_2d)
            torch.set_rng_state(state)
            label = self.transform(label)

            # Apply same transform to all slices of 2.5D input
            transformed_slices_2_5d = []
            for i in range(image_2_5d.shape[1]):
                slice_img = image_2_5d[:, i, :, :].unsqueeze(1) # shape (1, 1, H, W)
                torch.set_rng_state(state)
                transformed_slice = self.transform(slice_img)
                transformed_slices_2_5d.append(transformed_slice)
            image_2_5d = torch.cat(transformed_slices_2_5d, dim=1)

        return (
            (image_2d, image_2_5d),
            label,
            img_path.split(".nii")[0] + "_slice" + str(slice_idx) + ".nii",
        )