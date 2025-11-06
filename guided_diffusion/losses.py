"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np

import torch as th
import torch.nn as nn


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


class DiceLoss(nn.Module):
    """
    Computes the Dice loss, a common metric for segmentation tasks.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        """
        :param prediction: The model's output logits.
        :param target: The ground truth segmentation map.
        """
        prediction = th.sigmoid(prediction)
        iflat = prediction.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + self.smooth) /
                    (iflat.sum() + tflat.sum() + self.smooth))


class MedSegDiffV2Loss(nn.Module):
    """
    A composite loss function for the MedSegDiff-V2 model.

    This loss handles the two outputs of the model: the main segmentation
    and the uncertainty map ('cal'). It combines Dice loss, a BCE loss
    weighted by the uncertainty, and a regularization term for the uncertainty map.
    """
    def __init__(self, seg_loss_weight=1.0, cal_loss_weight=0.5, bce_loss_weight=1.0):
        super(MedSegDiffV2Loss, self).__init__()
        self.seg_loss_weight = seg_loss_weight
        self.cal_loss_weight = cal_loss_weight
        self.bce_loss_weight = bce_loss_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, main_output, cal_output, target):
        """
        :param main_output: The main segmentation logits from the diffusion model.
        :param cal_output: The uncertainty map (log variance) from the condition model.
        :param target: The ground truth segmentation map.
        """
        # Dice loss for the primary segmentation task
        dice = self.dice_loss(main_output, target)

        # Weighted BCE loss, modulated by the predicted uncertainty
        bce = self.bce_loss(main_output, target)
        
        # The uncertainty 'cal' is treated as log variance.
        # Weighting by exp(-cal) is equivalent to dividing by the variance,
        # reducing the loss contribution from uncertain pixels.
        weighted_bce = (bce * th.exp(-cal_output)).mean()

        # Regularization term for the uncertainty map to prevent it from
        # growing infinitely large and driving the weighted BCE loss to zero.
        cal_reg = cal_output.mean()

        total_loss = (self.seg_loss_weight * dice +
                      self.bce_loss_weight * weighted_bce +
                      self.cal_loss_weight * cal_reg)
        
        return total_loss
