import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import Tensor
from collections import OrderedDict
import re
import math
from typing import Type, Any, Callable, Union, List, Optional, cast, Tuple
from torch.distributions.uniform import Uniform
from einops import rearrange, repeat
from FreqFusion import FreqFusion
from thop import profile

up_kwargs = {'mode': 'bilinear', 'align_corners': True}
T_MAX = 512*64

from cuda import wkv

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv.backward(B, T, C,
                     w.float().contiguous(),
                     u.float().contiguous(),
                     k.float().contiguous(),
                     v.float().contiguous(),
                     gy.float().contiguous(),
                     gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


def mul_shift(input, shift_pixel=2, gamma=1/24):
    """
    对输入特征图进行24个方向的偏移操作，覆盖第一圈（8个）和第二圈（16个）相邻特征点
    :param input: 输入特征图，形状为 (B, C, H, W)
    :param shift_pixel: 基础偏移像素数（第二圈偏移以此为基础，默认2）
    :param gamma: 每个方向的通道占比（默认1/24，需≤1/24）
    :return: 偏移后的特征图
    """
    assert gamma <= 1/24, "gamma必须≤1/24以支持24个方向的分组"
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    group_size = int(C * gamma)  # 每组通道数
    s = shift_pixel  # 简化变量名，基础偏移量（第二圈主要用此值）
    s1 = 1  # 第一圈偏移量（固定为1）

    # -------------------------- 第一圈：8个相邻特征点（偏移1像素） --------------------------
    # 1. 水平右移 (→)
    output[:, 0:group_size, :, s1:W] = input[:, 0:group_size, :, 0:W-s1]
    # 2. 水平左移 (←)
    output[:, group_size:2*group_size, :, 0:W-s1] = input[:, group_size:2*group_size, :, s1:W]
    # 3. 垂直下移 (↓)
    output[:, 2*group_size:3*group_size, s1:H, :] = input[:, 2*group_size:3*group_size, 0:H-s1, :]
    # 4. 垂直上移 (↑)
    output[:, 3*group_size:4*group_size, 0:H-s1, :] = input[:, 3*group_size:4*group_size, s1:H, :]
    # 5. 右下对角线 (↘)
    output[:, 4*group_size:5*group_size, s1:H, s1:W] = input[:, 4*group_size:5*group_size, 0:H-s1, 0:W-s1]
    # 6. 左上对角线 (↖)
    output[:, 5*group_size:6*group_size, 0:H-s1, 0:W-s1] = input[:, 5*group_size:6*group_size, s1:H, s1:W]
    # 7. 左下对角线 (↙)
    output[:, 6*group_size:7*group_size, s1:H, 0:W-s1] = input[:, 6*group_size:7*group_size, 0:H-s1, s1:W]
    # 8. 右上对角线 (↗)
    output[:, 7*group_size:8*group_size, 0:H-s1, s1:W] = input[:, 7*group_size:8*group_size, s1:H, 0:W-s1]

    # -------------------------- 第二圈：16个相邻特征点（偏移2像素及组合） --------------------------
    # （一）正交方向扩展（纯水平/垂直，偏移2像素）
    # 9. 水平右移2像素 (→→)
    output[:, 8*group_size:9*group_size, :, s:W] = input[:, 8*group_size:9*group_size, :, 0:W-s]
    # 10. 水平左移2像素 (←←)
    output[:, 9*group_size:10*group_size, :, 0:W-s] = input[:, 9*group_size:10*group_size, :, s:W]
    # 11. 垂直下移2像素 (↓↓)
    output[:, 10*group_size:11*group_size, s:H, :] = input[:, 10*group_size:11*group_size, 0:H-s, :]
    # 12. 垂直上移2像素 (↑↑)
    output[:, 11*group_size:12*group_size, 0:H-s, :] = input[:, 11*group_size:12*group_size, s:H, :]

    # （二）对角线方向扩展（纯对角线，偏移2像素）
    # 13. 右下对角线2像素 (↘↘)
    output[:, 12*group_size:13*group_size, s:H, s:W] = input[:, 12*group_size:13*group_size, 0:H-s, 0:W-s]
    # 14. 左上对角线2像素 (↖↖)
    output[:, 13*group_size:14*group_size, 0:H-s, 0:W-s] = input[:, 13*group_size:14*group_size, s:H, s:W]
    # 15. 左下对角线2像素 (↙↙)
    output[:, 14*group_size:15*group_size, s:H, 0:W-s] = input[:, 14*group_size:15*group_size, 0:H-s, s:W]
    # 16. 右上对角线2像素 (↗↗)
    output[:, 15*group_size:16*group_size, 0:H-s, s:W] = input[:, 15*group_size:16*group_size, s:H, 0:W-s]

    # （三）混合方向（1像素+2像素组合，覆盖第二圈剩余8个方向）
    # 17. 右1+下2 (→↓↓)
    output[:, 16*group_size:17*group_size, s:H, s1:W] = input[:, 16*group_size:17*group_size, 0:H-s, 0:W-s1]
    # 18. 右2+下1 (→→↓)
    output[:, 17*group_size:18*group_size, s1:H, s:W] = input[:, 17*group_size:18*group_size, 0:H-s1, 0:W-s]
    # 19. 左1+下2 (←↓↓)
    output[:, 18*group_size:19*group_size, s:H, 0:W-s1] = input[:, 18*group_size:19*group_size, 0:H-s, s1:W]
    # 20. 左2+下1 (←←↓)
    output[:, 19*group_size:20*group_size, s1:H, 0:W-s] = input[:, 19*group_size:20*group_size, 0:H-s1, s:W]
    # 21. 右1+上2 (→↑↑)
    output[:, 20*group_size:21*group_size, 0:H-s, s1:W] = input[:, 20*group_size:21*group_size, s:H, 0:W-s1]
    # 22. 右2+上1 (→→↑)
    output[:, 21*group_size:22*group_size, 0:H-s1, s:W] = input[:, 21*group_size:22*group_size, s1:H, 0:W-s]
    # 23. 左1+上2 (←↑↑)
    output[:, 22*group_size:23*group_size, 0:H-s, 0:W-s1] = input[:, 22*group_size:23*group_size, s:H, s1:W]
    # 24. 左2+上1 (←←↑)
    output[:, 23*group_size:24*group_size, 0:H-s1, 0:W-s] = input[:, 23*group_size:24*group_size, s1:H, s:W]

    # 剩余通道不偏移
    output[:, 24*group_size:, ...] = input[:, 24*group_size:, ...]
    
    return output


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, init_mode='fancy', key_norm=False,
                 scan_schemes=None):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        attn_sz = n_embd
        self.device = None
        self.recurrence = 2
        self.scan_schemes = scan_schemes or [('top-left', 'horizontal'), ('bottom-right', 'vertical')]
        self.dwconv = nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False)
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)
        self.spatial_decay = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))
        self.spatial_first = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))

    # 实现四种方案的Z字形扫描
    def get_zigzag_indices(self, h, w, start='top-left', direction='horizontal'):
        indices = []
        if start == 'top-left':
            row_start = 0
            col_start = 0
            row_step = 1
            col_step = 1 if direction == 'horizontal' else 1
        elif start == 'top-right':
            row_start = 0
            col_start = w - 1
            row_step = 1
            col_step = -1 if direction == 'horizontal' else -1
        elif start == 'bottom-left':
            row_start = h - 1
            col_start = 0
            row_step = -1
            col_step = 1 if direction == 'horizontal' else 1
        elif start == 'bottom-right':
            row_start = h - 1
            col_start = w - 1
            row_step = -1
            col_step = -1 if direction == 'horizontal' else -1

        for i in range(h):
            current_row = row_start + row_step * i
            if direction == 'horizontal':
                if current_row % 2 == 0:
                    cols = list(range(w))
                else:
                    cols = list(range(w - 1, -1, -1))
                for col in cols:
                    indices.append(current_row * w + col)
            elif direction == 'vertical':
                if (col_start + col_step * i) % 2 == 0:
                    rows = list(range(h))
                else:
                    rows = list(range(h - 1, -1, -1))
                for row in rows:
                    indices.append(row * w + (col_start + col_step * i))
        return torch.tensor(indices, dtype=torch.long, device=self.device)


    def jit_func(self, x, resolution, scan_scheme):
        h, w = resolution
        start, direction = scan_scheme
        zigzag_order = self.get_zigzag_indices(h, w, start=start, direction=direction)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = mul_shift(x)  

        x = rearrange(x, 'b c h w -> b c (h w)')
        x = x[..., zigzag_order]
        x = rearrange(x, 'b c (h w) -> b (h w) c', h=h, w=w)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)
        return sr, k, v


    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device

        selected_scheme = self.scan_schemes[self.layer_id % len(self.scan_schemes)]
        sr, k, v = self.jit_func(x, resolution, selected_scheme)

        for j in range(self.recurrence):
            if j % 2 == 0:
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v) 
            else:

                h, w = resolution
                new_h, new_w = (h, w) if selected_scheme[1] == 'horizontal' else (w, h)
                zigzag_order = self.get_zigzag_indices(new_h, new_w, start=selected_scheme[0],
                                                       direction=selected_scheme[1])
                k = rearrange(k, 'b (h w) c -> b c h w', h=h, w=w)
                k = rearrange(k, 'b c h w -> b c (h w)')[..., zigzag_order]
                k = rearrange(k, 'b c (h w) -> b (h w) c', h=new_h, w=new_w)

                v = rearrange(v, 'b (h w) c -> b c h w', h=h, w=w)
                v = rearrange(v, 'b c h w -> b c (h w)')[..., zigzag_order]
                v = rearrange(v, 'b c (h w) -> b (h w) c', h=new_h, w=new_w)

                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
                k = rearrange(k, 'b (h w) c -> b (h w) c', h=h, w=w)
                v = rearrange(v, 'b (h w) c -> b (h w) c', h=h, w=w)

        x = v
        if self.key_norm is not None:
            x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4, init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)


    def forward(self, x, resolution):
        h, w = resolution
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = mul_shift(x) 
        x = rearrange(x, 'b c h w -> b (h w) c')
        k = self.key(x)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv

        return x


class Block(nn.Module):
    def __init__(self, outer_dim,layer_id, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.outer_norm1 = norm_layer(outer_dim)
        self.outer_attn = VRWKV_SpatialMix(n_embd=outer_dim, n_layer=None, layer_id=layer_id)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  
        self.outer_norm2 = norm_layer(outer_dim)
        self.outer_ffn = VRWKV_ChannelMix(n_embd=outer_dim, n_layer=None, layer_id=1)


    def forward(self, outer_tokens, H_out, W_out):
        
        B, N, C = outer_tokens.size()
        outer_patch_resolution = [H_out, W_out]
        outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens), outer_patch_resolution))
        outer_tokens = outer_tokens + self.drop_path(self.outer_ffn(self.outer_norm2(outer_tokens), outer_patch_resolution))
        return outer_tokens


class PatchMerging(nn.Module):
    def __init__(self, dim_in, dim_out, stride=2):
        super().__init__()
        self.stride = stride
        self.norm = nn.LayerNorm(dim_in)
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=2*stride-1, padding=stride-1, stride=stride),)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.conv(x)
        H, W = math.ceil(H / self.stride), math.ceil(W / self.stride)
        x = x.reshape(B, -1, H * W).transpose(1, 2)
        return x, H, W


class DAGC(nn.Module):
    def __init__(self, channels, kernel_size=7, factor=32):
        super(DAGC, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        assert channels // self.groups // 2 > 0
        self.pad = kernel_size // 2
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
        self.silu = nn.SiLU(inplace=True)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_hw = nn.AdaptiveAvgPool2d((1, 1))
        self.gn1 = nn.GroupNorm(min(16, channels//4), channels)
        self.gn2 = nn.GroupNorm(channels // self.groups // 2, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.conv1d = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=self.pad, groups=channels, bias=False)
        
    def forward(self, x):
        b, c, h, w = x.size()

        x_h = self.pool_h(x).view(b, c, h)
        x_h = self.sigmoid(self.gn1(self.conv1d(self.silu(self.conv1d(x_h))))).view(b, c, h, 1)

        x_w = self.pool_w(x).view(b, c, w)
        x_w = self.sigmoid(self.gn1(self.conv1d(self.silu(self.conv1d(x_w))))).view(b, c, 1, w)

        x_hw = x * x_h * x_w
        group_x = x_hw.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x1 = self.gn2(group_x)
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.pool_hw(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.pool_hw(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * self.sigmoid(weights)).reshape(b, c, h, w)


class ChannelBlock(nn.Module):
    def __init__(self, dim,  mlp_ratio=4., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False):
        super().__init__()
        self.ffn = ffn
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = DAGC(channels=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):

        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x_4d = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        
        cur = self.norm1(x).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        cur = self.attn(cur)
        cur = cur.permute(0, 2, 3, 1).contiguous().view(B, N, C)  # 转回3D
        
        x = x + self.drop_path(cur)
        
        # MLP处理
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.eca = eca_layer(out_features, 3)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = x.view(B, C, H, W)
        x = self.eca(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop(x)
        return x

class eca_layer(nn.Module):
    """Constructs a ECA module."""
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x) + x


class BasicBlock(nn.Module):
    def __init__(self,inplanes: int,planes: int,stride: int = 1,downsample: Optional[nn.Module] = None,groups: int = 1,
                 base_width: int = 64,dilation: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class CNNBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None, groups: int = 1,  base_width: int = 64, dilation: int = 1, 
                 norm_layer: Optional[Callable[..., nn.Module]] = None, mlp_ratio: float = 4., drop_path: float = 0., act_layer: nn.Module = nn.GELU):
        super(CNNBlock, self).__init__()

        self.basic_block = BasicBlock(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample, groups=groups, base_width=base_width, dilation=dilation, norm_layer=norm_layer)
        self.channel_block = ChannelBlock(dim=planes, mlp_ratio=mlp_ratio, drop_path=drop_path, act_layer=act_layer,
                                          norm_layer=nn.LayerNorm)
        
    def forward(self, x, H, W):

        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)

        x = self.basic_block(x)  # 输出形状: (B, C, H, W)
        
        # 转换形状以适配ChannelBlock（4D -> 3D: (B, N, C)，其中N=H*W）
        B, C, H, W = x.shape
        y = x.flatten(2).transpose(1, 2)  # 形状转换为: (B, H*W, C)  
        
        # 第二步：经过ChannelBlock处理
        y = self.channel_block(y)  # 输出形状: (B, N, C) 
        
        # 转换回4D特征图
        y = y.transpose(1, 2).view(B, C, H, W)  # 形状转换为: (B, C, H, W)
        x = x + y     
        x = x.flatten(2).transpose(1, 2)  # 转换为序列形状
        
        return x
    

class Stem(nn.Module):
    def __init__(self, img_size=512, in_chans=3, outer_dim=64):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.img_size = img_size

        self.num_patches = img_size[0] // 8 * img_size[1] // 8

        self.common_conv = nn.Sequential(
            nn.Conv2d(in_chans, 8, 3, stride=2, padding=1),  
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        
       
        self.outer_convs = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=2, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, outer_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(outer_dim),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.common_conv(x)

        H_out, W_out = H // 8, W // 8

        outer_tokens = self.outer_convs(x)
        outer_tokens = outer_tokens.permute(0, 2, 3, 1).reshape(B, H_out * W_out, -1)

        return outer_tokens, (H_out, W_out)


class Stage(nn.Module):
    def __init__(self, num_blocks, outer_dim, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        blocks = []
        drop_path = drop_path if isinstance(drop_path, list) else [drop_path] * num_blocks

        for j in range(num_blocks):
            blocks.append(Block(outer_dim, layer_id=j , drop_path=drop_path[j], norm_layer=norm_layer))

        self.blocks = nn.ModuleList(blocks)


    def forward(self, outer_tokens, H_out, W_out):
        for blk in self.blocks:
            outer_tokens = blk(outer_tokens, H_out, W_out)
        return outer_tokens


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.transposed_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.gelu1 = nn.GELU()
        self.conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = self.transposed_conv(x)
        x = self.batch_norm1(x)
        x = self.gelu1(x)
        x = self.conv(x)
        x = self.batch_norm2(x)
        x = self.gelu2(x)
        return x


class PyramidRiR_enc(nn.Module):
    def __init__(self, img_size=512, outer_dims=None, in_chans=3, drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        depths = [2, 4, 9, 2]
        cnn_depths = [1, 1, 1, 1] 
        drop_path_rates = 0.1  
        cnn_dpr = [x.item() for x in torch.linspace(0, drop_path_rates, sum(cnn_depths))]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.num_features = outer_dims[-1]

        self.patch_embed = Stem(img_size=img_size, in_chans=in_chans, outer_dim=outer_dims[0])
        num_patches = self.patch_embed.num_patches
        self.pos_embed_sentence = nn.Parameter(torch.zeros(1, num_patches, outer_dims[0]))
        self.interpolate_mode = 'bicubic'


        depth = 0
        self.sentence_merges = nn.ModuleList([])
        self.cnnblocks = nn.ModuleList([])
        self.stages = nn.ModuleList([])
        for i in range(4):
            if i > 0:
                self.sentence_merges.append(PatchMerging(outer_dims[i-1], outer_dims[i]))
            current_drop_path = cnn_dpr[sum(cnn_depths[:i])]
            self.cnnblocks.append(CNNBlock(outer_dims[i], outer_dims[i], stride = 1, norm_layer=nn.BatchNorm2d, mlp_ratio=4.0, drop_path=current_drop_path))
            self.stages.append(Stage(depths[i], outer_dim=outer_dims[i],drop_path=dpr[depth:depth + depths[i]], norm_layer=norm_layer))
            depth += depths[i]

        self.up_blocks = nn.ModuleList([])
        for i in range(4):
            if i < 3:
                self.up_blocks.append(UpsampleBlock(outer_dims[i], outer_dims[i+1]))
            else:
                self.up_blocks.append(UpsampleBlock(outer_dims[i], outer_dims[i]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'outer_pos', 'inner_pos'}

    def forward_features(self, x):
        outer_tokens, (H_out, W_out) = self.patch_embed(x)
        outputs = []

        for i in range(4):
            if i > 0:
                outer_tokens, H_out, W_out = self.sentence_merges[i - 1](outer_tokens, H_out, W_out)
            outer_tokens = self.cnnblocks[i](outer_tokens, H_out, W_out)
            outer_tokens = self.stages[i](outer_tokens, H_out, W_out)
            b, l, m = outer_tokens.shape
            mid_out = outer_tokens.reshape(b, int(math.sqrt(l)), int(math.sqrt(l)), m).permute(0, 3, 1, 2)
            mid_out = self.up_blocks[i](mid_out)
            outputs.append(mid_out)
        return outputs


    def forward(self, x):
        x = self.forward_features(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.ff = FreqFusion(in_channels, out_channels)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        x = self.ff(x2, x1)
        x = self.conv_bn_relu(x)
        return x
    

class UHH_RWKV(nn.Module):
    def __init__(self, channels, num_classes=2, img_size=512, in_chans=3):
        super(UHH_RWKV, self).__init__()

        self.RiR_backbone = PyramidRiR_enc(img_size=img_size, outer_dims=channels, in_chans=in_chans)
        self.decode4 = Decoder(channels[3], channels[3])
        self.decode3 = Decoder(channels[2], channels[2])
        self.decode2 = Decoder(channels[1], channels[1])
        self.decode0 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True), nn.Conv2d(channels[0], num_classes, kernel_size=1, bias=False))

    def forward(self, x):
        _, _, hei, wid = x.shape
        outputs = self.RiR_backbone(x)
        t1, t2, t3, t4 = outputs[0], outputs[1], outputs[2], outputs[3]
        d4 = self.decode4(t4, t3)
        d3 = self.decode3(d4, t2)
        d2 = self.decode2(d3, t1)
        out = self.decode0(d2)

        return out

if __name__ == "__main__":
    # 创建一个形状为 [batch_size, channels, height, width] 的虚拟输入张量
    # 注意：计算 FLOPs 时通常使用 batch_size=1 来评估单张图片的开销
    input = torch.randn(1, 3, 512, 512)
    print(f"初始输入形状: {input.shape}")

    # 初始化模块
    ela = UHH_RWKV(channels=[64, 128, 256, 512], num_classes=2, img_size=512, in_chans=3)
    ela = ela.cuda()  # 移到默认GPU（cuda:0）
    input = input.cuda()
    
    # --- 新增代码: 计算 FLOPs ---
    # profile 会自动遍历模型层级计算 FLOPs 和 Params
    # verbose=False 可以关闭 thop 默认的详细打印，只获取数值
    flops, params_thop = profile(ela, inputs=(input, ), verbose=False)
    
    print(f"FLOPs: {flops / 1e9:.2f} G")
    # ---------------------------

    # 前向传播
    output = ela(input)
    
    # 打印出输出张量的形状
    print(f"最终输出形状: {output.shape}")
    
    # 打印参数量 (保持你原有的逻辑，或者直接使用 thop 返回的 params)
    print(f"Params: {sum(p.numel() for p in ela.parameters())/1e6:.2f} M")

'''
if __name__ == "__main__":
    # 创建一个形状为 [batch_size, channels, height, width] 的虚拟输入张量
    input = torch.randn(1, 3, 512, 512)
    print(f"初始输入形状: {input.shape}")

    # 初始化模块
    ela = ZRiR(channels=[64, 128, 256, 512], num_classes=2, img_size=512, in_chans=3)
    ela = ela.cuda()  # 移到默认GPU（cuda:0）

    input = input.cuda()
    
    # 前向传播
    output = ela(input)
    # 打印出输出张量的形状，它将与输入形状相匹配。
    print(f"最终输出形状: {output.shape}")
    print(f"Params: {sum(p.numel() for p in ela.parameters())/1e6:.2f} M")
'''

'''
def force_move_all_tensors_to_cpu(module):
    """
    递归遍历模型的所有子模块，
    强制将所有属性中的 Tensor 移动到 CPU。
    """
    # 1. 处理子模块
    for child in module.children():
        force_move_all_tensors_to_cpu(child)
    
    # 2. 处理当前模块的属性 (self.xxx)
    # 这会找到像 self.sr 这样没有注册为 Parameter 的“隐形”张量
    for key, value in module.__dict__.items():
        if torch.is_tensor(value):
            if value.device.type != 'cpu':
                print(f"发现残留 GPU 张量: {key} (in {type(module).__name__})，正在强制移至 CPU...")
                setattr(module, key, value.cpu())

if __name__ == "__main__":
    # 1. 准备输入
    input = torch.randn(1, 3, 512, 512)
    print(f"初始输入形状: {input.shape}")

    # 2. 初始化模型
    # 注意：如果你的 __init__ 里有硬编码的 .cuda()，这里初始化完后，部分权重会在 GPU 上
    ela = ZRiR(channels=[64, 128, 256, 512], num_classes=2, img_size=512, in_chans=3)

    print("正在执行模型设备清理...")
    
    # 3. 标准操作：先尝试官方方法
    ela = ela.cpu()
    
    # 4. 【核心黑科技】：运行清理函数
    # 这步操作会从外部修正模型实例，把那个顽固的 'sr' 抓出来扔到 CPU 上
    force_move_all_tensors_to_cpu(ela)
    
    # 确保输入也在 CPU
    input = input.cpu()

    print("清理完毕，开始计算 FLOPs...")

    # 5. 计算 FLOPs (现在应该都在 CPU 上了)
    try:
        flops, params_thop = profile(ela, inputs=(input, ), verbose=False)
        
        flops_g = flops / 1e9
        print("-" * 30)
        print(f"Params: {params_thop / 1e6:.2f} M")
        print(f"FLOPs: {flops_g:.2f} G")
        print("-" * 30)
    except Exception as e:
        print(f"计算出错: {e}")
        print("建议尝试安装 'ptflops' 库，它比 thop 兼容性更好。")
'''