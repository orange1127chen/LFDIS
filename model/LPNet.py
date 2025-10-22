#!/usr/bin/python3
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.modules.mobilenetv2 import mobilenet_v2
from model.modules.DetailDe import DetailDe
from model.modules.BodyDe import BodyDe
from model.modules.MGCIDe import MGCIDe
from model.modules.weight_init import weight_init
from torchvision.ops import DeformConv2d
from operator import itemgetter
from axial_attention.reversible import ReversibleSequence
import numpy as np


def custom_complex_normalization(input_tensor, dim=-1):
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)

    normalized_tensor = torch.complex(norm_real, norm_imag)

    return normalized_tensor


# =============================================================================================

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道压缩
        b, c, _, _ = x.size()
        y = torch.mean(x, dim=(2, 3), keepdim=True)  # 全局平均池化
        y = y.view(b, c)  # 将输出展平
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y  # 按照计算得到的通道权重进行加权
    def initialize(self):
        weight_init(self)

# ==================================================================================
# helper functions

def exists(val):
    return val is not None

def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))

def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)

# calculates the permutation to bring the input tensor to something attend-able
# also calculates the inverse permutation to bring the tensor back to its original shape

def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)
      
    return permutations

# helper classes

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x)
            x = x + g(x)
        return x
 

class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape

        # merge all but axial dimension
        axial = axial.reshape(-1, t, d)

        # attention
        axial = self.fn(axial, **kwargs)

        # restore to original shape and permutation
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial



# axial pos emb

class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape, emb_dim_index = 1):
        super().__init__()
        parameters = []
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]

        self.num_axials = len(shape)

        for i, (axial_dim, axial_dim_index) in enumerate(zip(shape, ax_dim_indexes)):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = dim
            shape[axial_dim_index] = axial_dim
            parameter = nn.Parameter(torch.randn(*shape))
            setattr(self, f'param_{i}', parameter)

    def forward(self, x):
        for i in range(self.num_axials):
            x = x + getattr(self, f'param_{i}')
        return x


# attention

class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads = None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias = False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv = None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out


    
class AxialAttention(nn.Module):
    def __init__(self, dim, num_dimensions=2, heads=1, dim_heads=32, dim_index=1, sum_axial_out=True):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index >= 0 and dim_index < self.total_dimensions else 1  # Ensure valid dim_index

        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        # Ensure x has the correct number of dimensions
        # print("x shape:", x.shape)
        assert len(x.shape) == self.total_dimensions, f"Input tensor must have {self.total_dimensions} dimensions, but got {len(x.shape)}"
        
        # Check if dim_index is within valid range
        assert x.shape[self.dim_index] == self.dim, f"Expected dim {self.dim} at index {self.dim_index}, but got {x.shape[self.dim_index]}"

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out




class AGDC(nn.Module):
    def __init__(self, infeature):
        super(AGDC, self).__init__()
        self.arvix =AxialAttention(infeature)
        self.se_block = SEBlock(infeature)
        self.conv3_3 = nn.Conv2d(infeature, infeature, kernel_size=3, stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm2d(infeature)

    def forward(self, r, d):
        mul_fuse = r * d
        sa = self.arvix(mul_fuse)
        d_f = d * sa
        d_f = d + d_f
        d_ca = self.se_block(d_f)
        s = d * d_ca
        d_out = F.relu(self.bn3_3(self.conv3_3(s)), inplace=True)

        return d_out
    def initialize(self):
        weight_init(self)

# =======================================================================================
class LPNet(nn.Module):
    def __init__(self, cfg):
        super(LPNet, self).__init__()
        self.cfg = cfg


        self.convHR0 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=1), nn.Conv2d(16, 16, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.convHR1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1), nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.convHR2 = nn.Sequential(nn.Conv2d(24, 32, kernel_size=1), nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.convHR3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=1), nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.convHR4 = nn.Sequential(nn.Conv2d(96, 32, kernel_size=1), nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.convHR5 = nn.Sequential(nn.Conv2d(320, 32, kernel_size=1), nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(32), nn.ReLU(inplace=True))




        self.convLR1 = nn.Sequential(nn.Conv2d(16, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.convLR2 = nn.Sequential(nn.Conv2d(24, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.convLR3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.convLR4 = nn.Sequential(nn.Conv2d(96, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.convLR5 = nn.Sequential(nn.Conv2d(320, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.convDL1 = nn.Sequential(nn.Conv2d(16, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.convDL2 = nn.Sequential(nn.Conv2d(24, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.convDL3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.convDL4 = nn.Sequential(nn.Conv2d(96, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.convDL5 = nn.Sequential(nn.Conv2d(320, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
# ================================================================
 
        self.DT1 = AGDC(64)
        self.DT2 = AGDC(64)
        self.DT3 = AGDC(64)
        self.DT4 = AGDC(64)
        self.DT5 = AGDC(64)

        self.body = BodyDe()
        self.detail = DetailDe()
        self.mgcide = MGCIDe()

        self.linear_t = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linear_s = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.linear = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16),
                                    nn.ReLU(inplace=True), nn.Conv2d(16, 1, kernel_size=3, padding=1))

        if self.cfg is None or self.cfg.snapshot is None:
            weight_init(self)

        self.bkbone = mobilenet_v2()
        self.bkbone1 = mobilenet_v2()

        # self.Dbckbone = DepthBranch()
        self.bkbone.load_state_dict(torch.load('../LFDIS/mobilenet_v2-b0353104.pth'), strict=False)
        self.bkbone1.load_state_dict(torch.load('../LFDIS/mobilenet_v2-b0353104.pth'), strict=False)
        if self.cfg is not None and self.cfg.snapshot:
            print('load checkpoint')
            self.load_state_dict(torch.load(self.cfg.snapshot))
 

    def forward(self, x,z):

        y = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        z1 = z.repeat(1, 3, 1, 1)  
        y1 = F.interpolate(z1, size=(256, 256), mode='bilinear', align_corners=True)
        outHR0 = self.convHR0(x)

        outHR1, outHR2, outHR3, outHR4, outHR5 = self.bkbone(x)

        outLR1, outLR2, outLR3, outLR4, outLR5 = self.bkbone(y)
        outDL1, outDL2, outDL3, outDL4, outDL5= self.bkbone1(y1)
     
    #  =========================================================

        outHR1, outHR2, outHR3, outHR4, outHR5 = self.convHR1(outHR1), self.convHR2(outHR2), self.convHR3(
            outHR3), self.convHR4(outHR4), self.convHR5(outHR5)
        outLR1, outLR2, outLR3, outLR4, outLR5 = self.convLR1(outLR1), self.convLR2(outLR2), self.convLR3(
            outLR3), self.convLR4(outLR4), self.convLR5(outLR5)
        outDL1, outDL2, outDL3, outDL4, outDL5 = self.convDL1(outDL1), self.convDL2(outDL2), self.convDL3(
             outDL3), self.convDL4(outDL4), self.convDL5(outDL5)
        

       
        outLR1 = self.DT1(outLR1,outDL1)
        outLR2 = self.DT2(outLR2,outDL2)
        outLR3 = self.DT3(outLR3,outDL3)
        outLR4 = self.DT4(outLR4,outDL4)
        outLR5 = self.DT5(outLR5,outDL5)
        #  ==========================================================================================

    
        out_T32, out_T43, out_T54 = self.body([outLR5, outLR4, outLR3, outLR2, outLR1])

        out_S1, out_S2, out_S3, out_S4, out_S5, out_S6 = self.detail([outHR5, outHR4, outHR3, outHR2, outHR1, outHR0])
        maskFeature = self.mgcide([out_T32, out_T43, out_T54], [out_S1, out_S2, out_S3, out_S4, out_S5, out_S6])

        out_mask = self.linear(maskFeature)

        
        out_body = self.linear_t(out_T54)
        out_detail = self.linear_s(out_S6)

        return out_body, out_detail, out_mask