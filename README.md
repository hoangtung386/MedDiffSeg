# MedDiffSeg (Under Construction)

> **IMPORTANT NOTE**
>
> This is a **temporary** repository. This project is currently under active development and construction.

## Purpose

This repository is being created with the goal of comprehensively upgrading and improving upon the original **MedSegDiff** version.

The original repository can be found here:
[https://github.com/ImprintLab/MedSegDiff](https://github.com/ImprintLab/MedSegDiff)

## Comparison (Work in Progress)

The table below illustrates the (planned) differences between the original version and this new, upgraded version.

| MedSegDiff (Original Version) | MedDiffSeg (New Version) |
| :---: | :---: |
| **[ImprintLab/MedSegDiff](https://github.com/ImprintLab/MedSegDiff)** | **New Project (MedDiffSeg)** |
| ![MedSegDiff Image](./MedSegDiff_architecture.png) | ![MedDiffSeg Image](./MedDiffSeg_architecture.png) |

For training, run: 
> python scripts/segmentation_train.py --data_dir /home/admin1/Documents/data/training --out_dir ./output --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 8
---

*Please check back later for detailed updates. Thank you for your patience!*
