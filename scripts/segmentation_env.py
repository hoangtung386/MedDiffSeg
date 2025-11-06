import sys
import random
sys.path.append(".")
from guided_diffusion.utils import staple

import numpy
import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch.autograd import Function
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
import math
from PIL import Image
import matplotlib.pyplot as plt
from guided_diffusion.utils import staple
import argparse
import nibabel as nib

import collections
import logging
import math
import os
import time
from datetime import datetime

import dateutil.tz


def iou(outputs: np.array, labels: np.array):
    
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)


    return iou.mean()

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def eval_seg(pred,true_mask_p,threshold = (0.1, 0.3, 0.5, 0.7, 0.9)):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
            cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p [:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            iou_d += iou(disc_pred,disc_mask)
            iou_c += iou(cup_pred,cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()
            
        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    else:
        eiou, edice = 0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            eiou += iou(disc_pred,disc_mask)

            '''dice for torch'''
            edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            
        return eiou / len(threshold), edice / len(threshold)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="ISIC", help="Dataset name: ISIC or BRATS")
    parser.add_argument("--inp_pth", required=True, help="Path to prediction results")
    parser.add_argument("--out_pth", required=True, help="Path to ground truth data")
    parser.add_argument("--image_size", type=int, default=256, help="Image size for resizing")
    parser.add_argument("--debug", action='store_true', help="Enable debug prints and image saves")
    args = parser.parse_args()

    mix_res = (0, 0)
    num = 0
    pred_path = args.inp_pth
    gt_path = args.out_pth

    if args.data_name == 'ISIC':
        for root, dirs, files in os.walk(pred_path, topdown=False):
            for name in files:
                if 'ens' in name:
                    num += 1
                    ind = name.split('_')[0]
                    pred = Image.open(os.path.join(root, name)).convert('L')
                    gt_name = "ISIC_" + ind + "_Segmentation.png"
                    gt = Image.open(os.path.join(gt_path, gt_name)).convert('L')
                    
                    pred_tensor = torchvision.transforms.PILToTensor()(pred)
                    pred_tensor = torch.unsqueeze(pred_tensor, 0).float()
                    pred_tensor = pred_tensor / pred_tensor.max()

                    gt_tensor = torchvision.transforms.PILToTensor()(gt)
                    gt_tensor = torchvision.transforms.Resize((args.image_size, args.image_size))(gt_tensor)
                    gt_tensor = torch.unsqueeze(gt_tensor, 0).float() / 255.0
                    
                    if args.debug:
                        print(f"Processing ISIC sample: {ind}")
                        vutils.save_image(pred_tensor, fp=os.path.join('./results/', f'{ind}_pred.jpg'), nrow=1, padding=10)
                        vutils.save_image(gt_tensor, fp=os.path.join('./results/', f'{ind}_gt.jpg'), nrow=1, padding=10)

                    temp = eval_seg(pred_tensor, gt_tensor)
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
    
    elif args.data_name == 'BRATS':
        for root, dirs, files in os.walk(pred_path, topdown=False):
            for name in files:
                if 'ens' in name and 'jpg' in name: # e.g., BraTS20_Training_001_slice_100_output_ens.jpg
                    try:
                        num += 1
                        parts = name.replace('.jpg', '').split('_')
                        slice_num = int(parts[-3])
                        patient_id = '_'.join(parts[:-5])
                        
                        # Load prediction
                        pred = Image.open(os.path.join(root, name)).convert('L')
                        pred_tensor = torchvision.transforms.PILToTensor()(pred)
                        pred_tensor = torch.unsqueeze(pred_tensor, 0).float()
                        pred_tensor = pred_tensor / pred_tensor.max()

                        # Load ground truth
                        gt_filename = f"{patient_id}_seg.nii.gz"
                        # Assuming gt_path is the root training folder containing patient folders
                        gt_filepath = os.path.join(gt_path, patient_id, gt_filename)
                        
                        if not os.path.exists(gt_filepath):
                            print(f"Warning: Ground truth not found for {name}, skipping: {gt_filepath}")
                            num -= 1
                            continue

                        gt_vol = nib.load(gt_filepath).get_fdata()
                        gt_slice = gt_vol[:, :, slice_num] # Assuming H, W, D orientation
                        
                        # Create Whole Tumor mask
                        gt_wt = np.isin(gt_slice, [1, 2, 4]).astype(np.float32)
                        gt_tensor = torch.from_numpy(gt_wt).unsqueeze(0).unsqueeze(0)
                        gt_tensor = torchvision.transforms.Resize((args.image_size, args.image_size), antialias=True)(gt_tensor)

                        if args.debug:
                            print(f"Processing BRATS sample: {patient_id}, slice {slice_num}")
                            vutils.save_image(pred_tensor, fp=os.path.join('./results/', f'{patient_id}_{slice_num}_pred.jpg'), nrow=1, padding=10)
                            vutils.save_image(gt_tensor, fp=os.path.join('./results/', f'{patient_id}_{slice_num}_gt.jpg'), nrow=1, padding=10)

                        temp = eval_seg(pred_tensor, gt_tensor)
                        mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
                    except Exception as e:
                        print(f"Error processing file {name}: {e}")
                        num -=1
                        continue

    if num > 0:
        iou_score, dice_score = tuple([a / num for a in mix_res])
        print(f'Dataset: {args.data_name}')
        print(f'Processed {num} samples.')
        print(f'IOU: {iou_score}')
        print(f'Dice: {dice_score}')
    else:
        print("No samples were processed. Check your input paths and file names.")

if __name__ == "__main__":
    main()
