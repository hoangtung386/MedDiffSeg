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
from collections import OrderedDict
from prettytable import PrettyTable
#from mmcv.utils import print_log
# from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
# from mmseg.utils import get_root_logger

def eval(pre_eval_results):
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4
    total_area_intersect = sum(pre_eval_results[0]) # total_area_intersect.shape = (num_classes, )
    total_area_union = sum(pre_eval_results[1])     # total_area_union.shape = (num_classes, )
    total_area_pred_label = sum(pre_eval_results[2])
    total_area_label = sum(pre_eval_results[3])

    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label,)
    
    return ret_metrics


def total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label,nan_to_num=None,
                          beta=1):
    
    

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})

    iou = total_area_intersect / total_area_union
    acc = total_area_intersect / total_area_label
    dice = 2 * total_area_intersect / (
        total_area_pred_label + total_area_label)
    ret_metrics['IoU'] = iou
    #ret_metrics['Acc'] = acc
    #ret_metrics['Dice'] = dice

    precision = total_area_intersect / total_area_pred_label
    recall = total_area_intersect / total_area_label
    f_value = torch.tensor(
        [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
    ret_metrics['Fscore'] = f_value
    ret_metrics['Precision'] = precision
    ret_metrics['Recall'] = recall

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }

    return ret_metrics 


def pre_eval(pred, seg_map):
    pre_eval_results = []
    pre_eval_results.append(
        intersect_and_union(
            pred,
            seg_map,
            2))
    return pre_eval_results


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ):

    mask = (label != 255)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def f_score(precision, recall, beta=1):
    
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="ISIC", help="Dataset name: ISIC or BRATS")
    parser.add_argument("--inp_pth", required=True, help="Path to prediction results")
    parser.add_argument("--out_pth", required=True, help="Path to ground truth data")
    parser.add_argument("--image_size", type=int, default=256, help="Image size for resizing")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary segmentation")
    parser.add_argument("--debug", action='store_true', help="Enable debug prints and image saves")
    args = parser.parse_args()

    pred_path = args.inp_pth
    gt_path = args.out_pth
    results = []
    num_processed = 0

    if args.data_name == 'ISIC':
        class_names = ('background', 'lesion')
        for root, dirs, files in os.walk(pred_path, topdown=False):
            for name in files:
                if 'ens' in name and name.endswith('.jpg'):
                    try:
                        # e.g., ISIC_0000000_output_ens.jpg -> 0000000
                        img_id = name.split('_')[1]
                        
                        # Load prediction
                        pred = Image.open(os.path.join(root, name)).convert('L')
                        pred_tensor = torchvision.transforms.PILToTensor()(pred).float() / 255.0
                        
                        # Load ground truth
                        gt_name = f"ISIC_{img_id}_Segmentation.png"
                        gt = Image.open(os.path.join(gt_path, gt_name)).convert('L')
                        gt_tensor = torchvision.transforms.PILToTensor()(gt).float() / 255.0
                        gt_tensor = torchvision.transforms.Resize((args.image_size, args.image_size), antialias=True)(gt_tensor)

                        # Binarize and convert to integer labels
                        pred_labels = (pred_tensor > args.threshold).int()
                        gt_labels = (gt_tensor > 0.5).int()

                        results.extend(pre_eval(pred_labels, gt_labels))
                        num_processed += 1
                    except Exception as e:
                        print(f"Could not process {name}: {e}")

    elif args.data_name == 'BRATS':
        class_names = ('background', 'tumor')
        for root, dirs, files in os.walk(pred_path, topdown=False):
            for name in files:
                if 'ens' in name and name.endswith('.jpg'):
                    try:
                        # e.g., BraTS20_Training_001_slice_100_output_ens.jpg
                        parts = name.replace('.jpg', '').split('_')
                        slice_num = int(parts[-3])
                        patient_id = '_'.join(parts[:-5])

                        # Load prediction
                        pred = Image.open(os.path.join(root, name)).convert('L')
                        pred_tensor = torchvision.transforms.PILToTensor()(pred).float() / 255.0

                        # Load ground truth
                        gt_filename = f"{patient_id}_seg.nii.gz"
                        gt_filepath = os.path.join(gt_path, patient_id, gt_filename)
                        
                        if not os.path.exists(gt_filepath):
                            print(f"Warning: GT not found for {name}, skipping.")
                            continue

                        gt_vol = nib.load(gt_filepath).get_fdata()
                        gt_slice = gt_vol[:, :, slice_num]
                        
                        # Create Whole Tumor mask and convert to integer labels
                        gt_wt = np.isin(gt_slice, [1, 2, 4]).astype(np.int32)
                        gt_labels = torch.from_numpy(gt_wt).unsqueeze(0)
                        gt_labels = torchvision.transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.NEAREST)(gt_labels)

                        # Binarize prediction
                        pred_labels = (pred_tensor > args.threshold).int()

                        results.extend(pre_eval(pred_labels, gt_labels))
                        num_processed += 1
                    except Exception as e:
                        print(f"Could not process {name}: {e}")

    if not results:
        print("No samples were processed. Check paths and filenames.")
        return

    ret_metrics = eval(results)
    
    # summary table
    ret_metrics_summary = OrderedDict({
        ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })

    # each class table
    ret_metrics.pop('aAcc', None)
    ret_metrics_class = OrderedDict({
        ret_metric: np.round(ret_metric_value * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    ret_metrics_class.update({'Class': class_names})
    ret_metrics_class.move_to_end('Class', last=False)

    # for logger
    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items():
        class_table_data.add_column(key, val)

    summary_table_data = PrettyTable()
    for key, val in ret_metrics_summary.items():
        if key == 'aAcc':
            summary_table_data.add_column(key, [val])
        else:
            summary_table_data.add_column('m' + key, [val])

    print(f"Evaluation results for {args.data_name} ({num_processed} samples):")
    print('Per class results:')
    print('\n' + class_table_data.get_string())
    print('Summary:')
    print('\n' + summary_table_data.get_string())   
    

if __name__ == "__main__":
    main()
