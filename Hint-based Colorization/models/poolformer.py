# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PoolFormer implementation
"""
from multiprocessing import pool
import os
import copy
from unittest.mock import patch
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
import math
import torch.nn.functional as F

# try:
#     from mmseg.models.builder import BACKBONES as seg_BACKBONES
#     from mmseg.utils import get_root_logger
#     from mmcv.runner import _load_checkpoint
#     has_mmseg = True
# except ImportError:
#     print("If for semantic segmentation, please install mmsegmentation first")

# try:
#     from mmdet.models.builder import BACKBONES as det_BACKBONES
#     from mmdet.utils import get_root_logger
#     from mmcv.runner import _load_checkpoint
#     has_mmdet = True
# except ImportError:
#     print("If for detection, please install mmdetection first")

has_mmdet = False
has_mmseg = False


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'poolformer_s': _cfg(crop_pct=0.9),
    'poolformer_m': _cfg(crop_pct=0.95),
}

def _gumbel_sigmoid(
    logits, tau=1, hard=False, eps=1e-10, training = True, threshold = 0.5
):
    if training :
        # ~Gumbel(0,1)`
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        # Difference of two` gumbels because we apply a sigmoid
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
    else :
        y_soft = logits.sigmoid()

    if hard:
        # Straight through.
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).masked_fill(y_soft > threshold, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

def get_random_policy(policy, ratio):
    random_p = torch.empty_like(policy).fill_(ratio).bernoulli() + policy * 0.0  # add policy * 0.0 into the loop of loss calculation to avoid the DDP issue
    return random_p

class TokenSelectGate(nn.Module):
    def __init__(self, dim_in, tau=5, is_hard=True, threshold=0.5, bias=True, pre_softmax=True, mask_filled_value=float('-inf'), ada_token_nonstep=False, ada_token_detach_attn=True):
        super().__init__()
        self.count_flops = False
        self.ada_token_nonstep = ada_token_nonstep  # if using nonstep, no mlp_head is needed in each of these layers
        if not ada_token_nonstep:
            self.mlp_head = nn.Linear(dim_in, 1, bias=bias)
        self.norm = nn.Identity()
        self.is_hard = is_hard
        self.tau = tau
        self.threshold = threshold
        self.add_noise = True
        self.pre_softmax = pre_softmax
        self.mask_filled_value = mask_filled_value
        self.ada_token_detach_attn = ada_token_detach_attn
        self.random_policy = False
        self.random_token = False
        self.random_token_ratio = 1.

    def set_tau(self, tau):
        self.tau = tau

    def forward(self, x):
        """
        x shape      : (bs, length, c)
        y mask shape : (bs, length, 1)
        """
        b, l = x.shape[:2]

        # generate token policy step by step in each layer, including the first (couple of) blocks
        logits = self.mlp_head(self.norm(x))
        token_select = _gumbel_sigmoid(logits, self.tau, self.is_hard, threshold=self.threshold, training=self.training) # (bs, length, 1)
        # token_select = torch.cat([token_select.new_ones(b,1,1), token_select], dim=1) # 토큰 하나 빼놧던거 다시 합하는 과정인가?
        # token_select = token_select.unsqueeze(-1) #(b,l,1)
        # token_select = token_select.transpose(1,2) #(b,1,)


        # if self.count_flops :
        #     return attn, token_select.squeeze(1)

        # attn_policy = torch.bmm(token_select.transpose(-1,-2), token_select) #(b,l,l)
        # attn_policy = attn_policy.unsqueeze(1) #(b,1,l,l)
        # if self.ada_token_detach_attn :
        #     attn_policy = attn_policy.detach()

        # # use pre_softmax during inference in both pre-softmax or pre-softmax training
        # if self.pre_softmax or not self.training :
        #     eye_mat = attn.new_zeros((l,l))
        #     eye_mat = eye_mat.fill_diagonal_(1) #(1,1,l,l)
        #     attn = attn_pre_softmax * attn_policy + attn_pre_softmax.new_zeros(attn_pre_softmax.shape).masked_fill_((1 - attn_policy - eye_mat)>0, self.mask_filled_value)
        #     attn = attn.softmax(-1)
        #     assert not torch.isnan(attn).any(), 'token select pre softmax nan !'
        # else :
        #     attn = nn.functional.normalize(attn * attn_policy, 1, -1)

        return token_select

# STE version
class TokenSelectGateSTE(nn.Module):
    def __init__(self, dim_in, tau=5, is_hard=True, threshold=0.5, bias=True, pre_softmax=True, mask_filled_value=float('-inf'), ada_token_nonstep=False, ada_token_detach_attn=True):
        super().__init__()
        self.count_flops = False
        self.ada_token_nonstep = ada_token_nonstep  # if using nonstep, no mlp_head is needed in each of these layers
        self.norm = nn.Identity()
        self.is_hard = is_hard
        self.tau = tau
        self.threshold = threshold
        self.add_noise = True
        self.pre_softmax = pre_softmax
        self.mask_filled_value = mask_filled_value
        self.ada_token_detach_attn = ada_token_detach_attn
        self.random_policy = False
        self.random_token = False
        self.random_token_ratio = 1.

        #
        self.th = 0
        self.threshold = nn.Parameter(self.th * torch.ones(1, 1, dim_in), requires_grad=True) # input shape : (bs, length, c)
        self.step = BinaryStep.apply
        # gumbel
        self.tau = 5
        self.is_hard = True
        self.gumbel_threshold = 0.5

        # init
        with torch.no_grad():
            self.threshold.data.fill_(0.)

    def set_tau(self, tau):
        self.tau = tau

    def forward(self, x):
        """
        x shape      : (bs, length, c)
        y mask shape : (bs, length, 1)
        """
        b, l = x.shape[:2]

        # generate token policy step by step in each layer, including the first (couple of) blocks
        token_select = (x.abs() - self.threshold).mean(2).unsqueeze(2)
        token_select = self.step(token_select)


        # logits = self.mlp_head(self.norm(x))
        # token_select = _gumbel_sigmoid(logits, self.tau, self.is_hard, threshold=self.threshold, training=self.training) # (bs, length, 1)

        return token_select

class BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input>0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2-4*torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input*additional


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class CnnHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.head = nn.Conv2d(embed_dim, 512, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.up = nn.Upsample(size=14)
    def forward(self, x):
        # input : (bs, c, h, w)
        x = self.up(x)
        x = self.head(x)
        x = x.flatten(2).permute(0,2,1)
        # x = rearrange(x, 'b c p1 p2 -> b (p1 p2) c')
        return x


class LocalAttentionHead(nn.Module):
    def __init__(
            self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, use_rpb=False, window_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        # masking attn
        mask = torch.ones((window_size**2, window_size**2))
        kernel_size = 3
        for i in range(window_size):
            for j in range(window_size):
                cur_map = torch.ones((window_size, window_size))
                stx, sty = max(i - kernel_size // 2, 0), max(j - kernel_size // 2, 0)
                edx, edy = min(i + kernel_size // 2, window_size - 1), min(j + kernel_size // 2, window_size - 1)
                cur_map[stx:edx + 1, sty:edy + 1] = 0
                cur_map = cur_map.flatten()
                mask[i * window_size + j] = cur_map
        self.register_buffer('mask', mask)

        # relative positional bias option
        self.use_rpb = use_rpb
        if use_rpb:
            self.window_size = window_size
            self.rpb_table = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
            trunc_normal_(self.rpb_table, std=.02)

            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, h, w
            coords_flatten = torch.flatten(coords, 1)  # 2, h*w
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, h*w, h*w
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # h*w, h*w, 2
            relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)  # h*w, h*w
            self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # masking attn
        mask_value = max_neg_value(attn)
        attn.masked_fill_(self.mask.bool(), mask_value)

        if self.use_rpb:
            relative_position_bias = self.rpb_table[self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # h*w,h*w,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, h*w, h*w
            attn += relative_position_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv. 
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0, 
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.flops = 0
        self.patch_size = patch_size[0]

    def forward(self, x):
        bs, c1,h,w = x.shape
        x = self.proj(x)
        x = self.norm(x)

        # bs, c,h,w = x.shape
        # self.flops += c1*c*h*w*self.patch_size*self.patch_size
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
        
        self.k = pool_size
        self.flops= 0 

    def forward(self, x):
        y = self.pool(x) - x
        bs, c, h, w = y.shape

        self.flops += (self.k * self.k -1) * c * h * w 

        return y

"""original"""
class Mlp3(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
        self.flops = 0

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(0, 2, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # bs, c2, h2, w2 = x.shape
        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0, 2, 1).view(bs, c, h, w)

        # bs, c3, h3, w3 = x.shape

        # self.flops += h2*w2*1*1*c*c2 + h3*w3*1*1*c2*c3
        return x, 0

"""masking"""
class Mlp2(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.gate = TokenSelectGate(dim_in=in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(0, 2, 1)
        mask = self.gate(x)
        x = x*mask
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = x*mask
        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0, 2, 1).view(bs, c, h, w)
        return x, mask

"""temp"""
class Mlp2(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    # 320 gae
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.gate = TokenSelectGate(dim_in=in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(0, 2, 1)
        x2 = x
        mask = self.gate(x)
        # x = x*mask
        x = x[:,::4,:]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # x = x*mask
        x = x[:,::4,:]
        x = self.fc2(x)
        x = self.drop(x)
        # x = x.permute(0, 2, 1).view(bs, c, h, w)
        x = x2.permute(0, 2, 1).view(bs, c, h, w)
        return x, mask

class PrunedLinear(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(PrunedLinear, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.weight = nn.Parameter(torch.Tensor(out_planes, in_planes))
        self.bias = nn.Parameter(torch.Tensor(out_planes))

        # self.mlp_head = nn.Linear(in_planes, in_planes, bias=bias)

        # threshold (channel pruning?)
        self.th = 0
        self.threshold = nn.Parameter(self.th * torch.ones(out_planes, 1), requires_grad=True) # weight shape : (N, C)
        self.step = BinaryStep.apply
        # gumbel
        self.tau = 5
        self.is_hard = True
        self.gumbel_threshold = 0.5

        # init
        with torch.no_grad():
            self.threshold.data.fill_(0.)

        # self.apply(self._init_weights)

    def _init_weights(self):
        trunc_normal_(self.weight, std=.02)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        """
        wegiht shape         : (out_c, in_c)
        threshold shape      : (1, in_c)
        channel_select shape : (out_c, in_c)

        """
        mask = (self.weight.abs() - self.threshold).mean(dim=0).unsqueeze(0)
        mask = self.step(mask)
        # mask = _gumbel_sigmoid(mask, self.tau, self.is_hard, threshold=self.gumbel_threshold, training=self.training)
        
        # print('mask', mask.shape)
        # print('weight', self.weight.shape)

        masked_weight = self.weight * mask 
        output = torch.nn.functional.linear(x, masked_weight, self.bias)

        return output, mask

"""token + channel"""
class Mlp2(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = PrunedLinear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = PrunedLinear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.gate = TokenSelectGateSTE(dim_in=in_features)

        # init weights
        # self.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(0, 2, 1)
        t_mask = self.gate(x)
        x = x*t_mask
        x, c_mask = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = x*t_mask
        x, c_mask2 = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0, 2, 1).view(bs, c, h, w)
        return x, t_mask, (c_mask.mean().view(1), c_mask2.mean().view(1))

class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth, 
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.token_list = []
        self.channel_list = []

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))

            output, token_mask  = self.mlp(self.norm2(x))
            self.token_list.append(token_mask)

            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * output)
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# token & channel
class PoolFormerBlock2(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth, 
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.token_list = []
        self.channel_list = []

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))

            output, token_mask, (channel_mask, channel_mask2)  = self.mlp(self.norm2(x))
            self.token_list.append(token_mask)
            self.channel_list.append(channel_mask)
            self.channel_list.append(channel_mask2)

            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * output)
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers, 
                 pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop_rate=.0, drop_path_rate=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks 
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PoolFormerBlock(
            dim, pool_size=pool_size, mlp_ratio=mlp_ratio, 
            act_layer=act_layer, norm_layer=norm_layer, 
            drop=drop_rate, drop_path=block_dpr, 
            use_layer_scale=use_layer_scale, 
            layer_scale_init_value=layer_scale_init_value, 
            ))
    blocks = nn.Sequential(*blocks)

    return blocks


class PoolFormer(nn.Module):
    """
    PoolFormer, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, --pool_size: the embedding dims, mlp ratios and 
        pooling size for the 4 stages
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalization and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad: 
        specify the downsample (patch embed.)
    --fork_feat: whether output features of the 4 stages, for dense prediction
    --init_cfg, --pretrained: 
        for mmdetection and mmsegmentation to load pretrained weights
    """
    def __init__(self, layers, embed_dims=None, 
                 mlp_ratios=None, downsamples=None, 
                 pool_size=3, 
                 norm_layer=GroupNorm, act_layer=nn.GELU, 
                 num_classes=1000,
                 in_patch_size=7, in_stride=4, in_pad=2, 
                 down_patch_size=3, down_stride=2, down_pad=1, 
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5, 
                 fork_feat=False,
                 init_cfg=None, 
                 pretrained=None, 
                 **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.in_chans = 4

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad, 
            in_chans= self.in_chans, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, 
                                 pool_size=pool_size, mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer, 
                                 drop_rate=drop_rate, 
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale, 
                                 layer_scale_init_value=layer_scale_init_value)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride, 
                        padding=down_pad, 
                        in_chans=embed_dims[i], embed_dim=embed_dims[i+1]
                        )
                    )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        # Colorization module
        # if head_mode == 'linear':
        #     self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # elif head_mode == 'cnn':
        self.head = CnnHead(embed_dims[-1], num_classes)
        # elif head_mode == 'locattn':
        #     self.head = LocalAttentionHead(embed_dim, num_classes, window_size=img_size // patch_size)
        # else:
        #     raise NotImplementedError('Check head type')

        self.tanh = nn.Tanh()
        #
        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model 
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading 
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)
            
            # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x, mask):
    # def forward(self, x):

        # colorization
        # mask is 1D of 2D if 2D
        B, _, H, W = x.shape
        assert mask.dim() == 2, f'Check the mask dimension mask.dim() == 2 but {mask.dim()}.'

        _, L = mask.shape
        # assume square inputs
        hint_size = int(math.sqrt(H * W // L))
        _device = '.cuda' if x.device.type == 'cuda' else ''

        # gh

        # hint location = 0, non-hint location = 1
        mask = torch.reshape(mask, (B, H // hint_size, W // hint_size))
        _mask = mask.unsqueeze(1).type(f'torch{_device}.FloatTensor')
        _full_mask = F.interpolate(_mask, scale_factor=hint_size)  # Needs to be Float
        full_mask = _full_mask.type(f'torch{_device}.BoolTensor')

        # mask ab channels
        # x[:, 1, :, :].masked_fill_(full_mask.squeeze(1), 0)
        # x[:, 2, :, :].masked_fill_(full_mask.squeeze(1), 0)
        _avg_x = F.interpolate(x, size=(H // hint_size, W // hint_size), mode='bilinear')
        _avg_x[:, 1, :, :].masked_fill_(mask.squeeze(1), 0)
        _avg_x[:, 2, :, :].masked_fill_(mask.squeeze(1), 0)
        x_ab = F.interpolate(_avg_x, scale_factor=hint_size, mode='nearest')[:, 1:, :, :]
        x = torch.cat((x[:, 0, :, :].unsqueeze(1), x_ab), dim=1)

        if self.in_chans == 4:
            x = torch.cat((x, 1 - _full_mask), dim=1)


        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        # cls_out = self.head(x.mean([-2, -1]))
        cls_out = self.head(x)

        # for image classification
        return cls_out


model_urls = {
    "poolformer_s12": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s12.pth.tar",
    "poolformer_s24": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s24.pth.tar",
    "poolformer_s36": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s36.pth.tar",
    "poolformer_m36": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m36.pth.tar",
    "poolformer_m48": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m48.pth.tar",
}


@register_model
def poolformer_s12(pretrained=False, **kwargs):
    """
    PoolFormer-S12 model, Params: 12M
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios: 
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_s']
    # if pretrained:
    url = model_urls['poolformer_s12']
    checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
    # model.load_state_dict(checkpoint)

    from collections import OrderedDict
    new_dict = OrderedDict()

    for i, j in checkpoint.items():
        if 'mlp.fc1.weight' in i :
            j = j.squeeze(2).squeeze(2)
        if 'mlp.fc2.weight' in i:
            j = j.squeeze(2).squeeze(2)

        if 'patch_embed.proj.weight' ==i:
            continue

        new_dict[i] = j

    # model.load_state_dict(torch.load('./saved_models/ffhq_else224/poolformer_s12/best/checkpoint-206.pth')['model'], strict=False)
    # model.load_state_dict(new_dict, strict=False)
    return model


@register_model
def poolformer_s24(pretrained=False, **kwargs):
    """
    PoolFormer-S24 model, Params: 21M
    """
    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_s']
    if pretrained:
        url = model_urls['poolformer_s24']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def poolformer_s36(pretrained=False, **kwargs):
    """
    PoolFormer-S36 model, Params: 31M
    """
    layers = [6, 6, 18, 6]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        layer_scale_init_value=1e-6, 
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_s']
    if pretrained:
        url = model_urls['poolformer_s36']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)

        from collections import OrderedDict
        new_dict = OrderedDict()

        for i, j in checkpoint.items():
            if 'mlp.fc1.weight' in i :
                j = j.squeeze(2).squeeze(2)
            if 'mlp.fc2.weight' in i:
                j = j.squeeze(2).squeeze(2)

            new_dict[i] = j

        model.load_state_dict(new_dict)

    
    return model


@register_model
def poolformer_m36(pretrained=False, **kwargs):
    """
    PoolFormer-M36 model, Params: 56M
    """
    layers = [6, 6, 18, 6]
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        layer_scale_init_value=1e-6, 
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_m']
    if pretrained:
        url = model_urls['poolformer_m36']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def poolformer_m48(pretrained=False, **kwargs):
    """
    PoolFormer-M48 model, Params: 73M
    """
    layers = [8, 8, 24, 8]
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        layer_scale_init_value=1e-6, 
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_m']
    if pretrained:
        url = model_urls['poolformer_m48']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model

