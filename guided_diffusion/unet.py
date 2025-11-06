from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from copy import deepcopy
from .utils import softmax_helper,sigmoid_helper
from .utils import InitWeights_He
from batchgenerators.augmentations.utils import pad_nd_image
from .utils import no_op
from .utils import to_cuda, maybe_to_torch
from scipy.ndimage.filters import gaussian_filter
from typing import Union, Tuple, List
from torch.cuda.amp import autocast
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    layer_norm,
)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
        )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        # dw
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        # pw
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class MobBlock(nn.Module):
    def __init__(self,ind):
        super().__init__()


        if ind == 0:
            self.stage = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 1),
            conv_dw(128, 128, 1)
        )
        elif ind == 1:
            self.stage  = nn.Sequential(
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1)
        )
        elif ind == 2:
            self.stage = nn.Sequential(
            conv_dw(256, 256, 2),
            conv_dw(256, 256, 1)
            )
        else:
            self.stage = nn.Sequential(
                conv_dw(256, 512, 2),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1)
            )

    def forward(self,x):
        return self.stage(x)



class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class FFParser(nn.Module):
    def __init__(self, dim, h=128, w=65):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, C, H, W = x.shape
        assert H == W, "height and width are not equal"
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size

        # x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')

        x = x.reshape(B, C, H, W)

        return x


class ConvNormNonlin(nn.Module):
    def __init__(self, input_channels, output_channels, conv_op=nn.Conv2d,
                 conv_kwargs={'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True},
                 norm_op=nn.BatchNorm2d, norm_op_kwargs={'eps': 1e-5, 'affine': True, 'momentum': 0.1},
                 nonlin=nn.LeakyReLU, nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True}):
        super(ConvNormNonlin, self).__init__()
        self.conv = conv_op(input_channels, output_channels, **conv_kwargs)
        self.norm = norm_op(output_channels, **norm_op_kwargs) if norm_op is not None else None
        self.nonlin = nonlin(**nonlin_kwargs) if nonlin is not None else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.nonlin is not None:
            x = self.nonlin(x)
        return x

class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs={'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True},
                 norm_op=nn.BatchNorm2d, norm_op_kwargs={'eps': 1e-5, 'affine': True, 'momentum': 0.1},
                 dropout_op=nn.Dropout2d, dropout_op_kwargs={'p': 0.5, 'inplace': True},
                 nonlin=nn.LeakyReLU, nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True},
                 basic_block=ConvNormNonlin):
        super(StackedConvLayers, self).__init__()
        self.convs = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, conv_op, conv_kwargs,
                           norm_op, norm_op_kwargs, nonlin, nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, conv_op, conv_kwargs,
                           norm_op, norm_op_kwargs, nonlin, nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.convs(x)

class Generic_UNet(nn.Module):
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=False, dropout_in_localization=False,
                 final_nonlin=lambda x: x, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None, upscale_logits=False, convolutional_pooling=True,
                 convolutional_upsampling=True, max_num_features=None, basic_block=ConvNormNonlin,
                 seg_output_use_bias=False, highway=False, anchor_out=False):
        super(Generic_UNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.nonlin = nonlin
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.highway = highway
        self.anchor_out = anchor_out

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
        self.norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        self.dropout_op_kwargs = {'p': 0, 'inplace': True}
        self.nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            # Downsampling path
            self.conv_kwargs['kernel_size'] = 3
            self.conv_kwargs['padding'] = 1
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              basic_block=basic_block))
            if self.convolutional_pooling:
                pool_op = self.conv_op(output_features, output_features, 2, 2, bias=False)
            else:
                pool_op = nn.MaxPool2d(2) if self.conv_op == nn.Conv2d else nn.MaxPool3d(2)
            self.td.append(pool_op)
            input_features = output_features
            output_features = int(round(output_features * feat_map_mul_on_downscale))

        # Bottleneck
        self.conv_kwargs['kernel_size'] = 3
        self.conv_kwargs['padding'] = 1
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                              self.nonlin, self.nonlin_kwargs, basic_block=basic_block)))

        # Upsampling path
        for u in range(num_pool):
            if self.convolutional_upsampling:
                transpconv = nn.ConvTranspose2d if self.conv_op == nn.Conv2d else nn.ConvTranspose3d
                self.tu.append(transpconv(output_features, output_features // 2, 2, 2, bias=False))
            else:
                self.tu.append(nn.Upsample(scale_factor=2, mode='bilinear' if self.conv_op == nn.Conv2d else 'trilinear'))
            
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(output_features, output_features // 2, num_conv_per_stage -1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(output_features // 2, output_features // 2, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        # Final convolution
        self.final_conv = nn.Conv2d(output_features // 2, num_classes, 1) if conv_op == nn.Conv2d else nn.Conv3d(output_features // 2, num_classes, 1)
        self.final_nonlin = final_nonlin

    def forward(self, x):
        skips = []
        
        # Encoder path
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            x = self.td[d](x)

        # Bottleneck
        x = self.conv_blocks_context[-1](x)
        bottleneck_features = x

        # Decoder path
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips.pop()), dim=1)
            x = self.conv_blocks_localization[u](x)

        seg_output = self.final_conv(x)

        # The calls in UNetModel_MedSegDiffV2 expect two return values.
        return self.final_nonlin(seg_output), bottleneck_features


class NBP_Filter(nn.Module):
    def __init__(self, channel, h=128, w=65):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(channel, h, w, 2, dtype=torch.float32) * 0.02)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        return x

class SS_Former(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        
        self.norm = normalization(in_channels)
        self.q_conv = conv_nd(2, in_channels, in_channels, 1)
        self.k_conv = conv_nd(2, in_channels, in_channels, 1)
        self.v_conv = conv_nd(2, in_channels, in_channels, 1)
        
        self.nbp_filter = NBP_Filter(in_channels)
        self.proj_out = zero_module(conv_nd(2, in_channels, out_channels, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.ReLU(),
            nn.Linear(in_channels * 4, in_channels)
        )

    def forward(self, x, anchor_cond, semantic_cond):
        b, c, h, w = x.shape

        # q from diffusion U-Net bottleneck (x),
        # k from 2.5D semantic condition (semantic_cond),
        # v from 2D anchor condition (anchor_cond)
        q = self.q_conv(x)
        k = self.k_conv(semantic_cond)
        v = self.v_conv(anchor_cond)

        # Apply NBP-Filter
        q_fft = self.nbp_filter(q)
        k_fft = self.nbp_filter(k)

        # Attention
        scale = 1 / math.sqrt(c)
        weight = torch.einsum("bchw,bchw->bhw", q_fft * scale, k_fft * scale)
        weight = torch.softmax(weight.view(b, -1), dim=-1).view(b, h, w)

        attn = torch.einsum("bhw,bchw->bchw", weight, v)

        # MLP
        attn = attn.permute(0, 2, 3, 1)  # B, H, W, C
        mlp_out = self.mlp(attn)
        mlp_out = mlp_out.permute(0, 3, 1, 2)  # B, C, H, W

        # Add a residual connection to preserve the diffusion path features
        return self.proj_out(mlp_out) + x

class SymmetryEnhancedAttention(nn.Module):
    """
    Symmetry Enhanced Attention (SEA) module.
    As described in the MedSegDiff-V2 diagram.
    This is an exemplary implementation based on the diagram.
    It assumes 3D features and performs self-attention and symmetry-attention.
    """
    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def _forward(self, x):
        # Assuming x is a 3D feature map (B, C, D, H, W)
        b, c, d, h, w = x.shape
        x_flat = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x_flat))

        # Self-attention
        self_h = self.attention(qkv)

        # Symmetry-attention
        # Assuming symmetry along depth axis (D), which is the slice axis
        x_flipped = th.flip(x, dims=[2])
        x_flipped_flat = x_flipped.reshape(b, c, -1)
        qkv_flipped = self.qkv(self.norm(x_flipped_flat))

        q, _, _ = th.chunk(qkv, 3, dim=1)
        _, k_flipped, v_flipped = th.chunk(qkv_flipped, 3, dim=1)
        
        qkv_sym = th.cat([q, k_flipped, v_flipped], dim=1)
        sym_h = self.attention(qkv_sym)

        # The diagram shows addition, so we add them.
        h = self_h + sym_h
        h = self.proj_out(h)
        
        return (x_flat + h).reshape(b, c, d, h, w)

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

class UNetModel_MedSegDiffV2(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    This version corresponds to MedSegDiff-V2, including a 2.5D conditioning U-Net.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # --- Diffusion U-Net Path ---
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
        
        self.ss_former = SS_Former(ch, ch)

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, time_embed_dim, dropout, out_channels=model_channels * mult, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads_upsample, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        # --- Conditioning Models ---
        self.hwm = Generic_UNet(self.in_channels - 1, 32, ch, 4, anchor_out=True)
        
        self.encoder_2_5d = Generic_UNet(1, 16, ch, 4, conv_op=nn.Conv3d, norm_op=nn.InstanceNorm3d, num_conv_per_stage=1)
        self.decoder_2d = Generic_UNet(ch, 16, ch, 4, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, num_conv_per_stage=1)
        self.sea = SymmetryEnhancedAttention(ch)
        self.cal_head = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, ch, 1, 1)
        )

    def forward(self, x, timesteps, y=None, x_2_5d=None):
        assert x_2_5d is not None, "MedSegDiff-V2 requires a 2.5D input `x_2_5d`"
        
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        # --- Conditioning Path ---
        # 2.5D Encoder
        _, features_3d = self.encoder_2_5d(x_2_5d.type(self.dtype))
        
        # SEA
        sea_features_3d = self.sea(features_3d)
        
        # Extract center slice and pass to 2D Decoder
        center_slice_idx = sea_features_3d.shape[2] // 2
        sea_features_2d = sea_features_3d[:, :, center_slice_idx, :, :]
        semantic_cond, _ = self.decoder_2d(sea_features_2d)

        # 2D Condition Model for Anchor Condition
        anchor_cond, _ = self.hwm(x[:, :-1, ...])
        
        # --- Compute 'cal' output ---
        cal = self.cal_head(anchor_cond)

        # --- Diffusion Path ---
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        
        # Bottleneck with SS-Former
        h = self.ss_former(h, anchor_cond, semantic_cond)

        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
            
        h = h.type(x.dtype)
        main_output = self.out(h)
        
        return main_output, cal

class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For encoding images into latent representations.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # --- Encoder UNet Path ---
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        self.encoder = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, y=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        # --- Encoding Path ---
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
        
        return self.encoder(h)






