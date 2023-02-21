import json
import math
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.cuda import amp

from utils.datasets import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, time_sync

from torch.autograd import Variable
import torch.nn.functional as F


class MaskConv2d(torch.nn.Module):
    """
    Mask Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d+activation
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(MaskConv2d, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class TIEModule(nn.Module):
    def __init__(self, c1, c2):
        super(TIEModule, self).__init__()
        """
        未进行归一化
        """
        # 生成float型数组
        kernel_const_hori = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype='float32')
        # GPU上的数据类型
        kernel_const_hori = torch.FloatTensor(kernel_const_hori).unsqueeze(0)
        # 赋予算子卷积权重
        self.weight_const_hori = nn.Parameter(data=kernel_const_hori, requires_grad=False)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.delta = nn.Parameter(torch.zeros(1))
        kernel_const_vertical = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype='float32')
        kernel_const_vertical = torch.FloatTensor(kernel_const_vertical).unsqueeze(0)
        self.weight_const_vertical = nn.Parameter(data=kernel_const_vertical, requires_grad=False)
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()

        self.conv2d_1_attention = torch.nn.Sequential()
        self.conv2d_1_attention.add_module('conv2d_1_attention', nn.Conv2d(2, 8, kernel_size=3, padding=1))
        self.bn_edge_1 = nn.BatchNorm2d(8)
        self.conv2d_2_attention = torch.nn.Sequential()
        self.conv2d_2_attention.add_module('conv2d_2_attention', nn.Conv2d(8, 16, kernel_size=3, padding=1))
        self.bn_edge_2 = nn.BatchNorm2d(16)
        self.conv2d_3_attention = torch.nn.Sequential()
        self.conv2d_3_attention.add_module('conv2d_3_attention', nn.Conv2d(16, 16, kernel_size=3, padding=1))
        self.bn_edge_3 = nn.BatchNorm2d(16)
        self.conv2d_4_attention = torch.nn.Sequential()
        self.conv2d_4_attention.add_module('conv2d_4_attention', nn.Conv2d(16, 8, kernel_size=3, padding=1))
        self.bn_edge_4 = nn.BatchNorm2d(8)
        self.conv2d_5_attention = torch.nn.Sequential()
        self.conv2d_5_attention.add_module('conv2d_5_attention', nn.Conv2d(8, 1, kernel_size=3, padding=1))
        self.bn_edge_5 = nn.BatchNorm2d(1)

        self.conv2d_1_rgb_attention = torch.nn.Sequential()
        self.conv2d_1_rgb_attention.add_module('conv2d_1_rgb_attention', nn.Conv2d(4, 8, kernel_size=3, padding=1))
        self.bn_rgb_1 = nn.BatchNorm2d(8)
        self.conv2d_2_rgb_attention = torch.nn.Sequential()
        self.conv2d_2_rgb_attention.add_module('conv2d_2_rgb_attention', nn.Conv2d(8, 16, kernel_size=3, padding=1))
        self.bn_rgb_2 = nn.BatchNorm2d(16)
        self.conv2d_3_rgb_attention = torch.nn.Sequential()
        self.conv2d_3_rgb_attention.add_module('conv2d_3_rgb_attention', nn.Conv2d(16, 16, kernel_size=3, padding=1))
        self.bn_rgb_3 = nn.BatchNorm2d(32)
        self.conv2d_4_rgb_attention = torch.nn.Sequential()
        self.conv2d_4_rgb_attention.add_module('conv2d_4_rgb_attention', nn.Conv2d(32, 16, kernel_size=3, padding=1))
        self.bn_rgb_4 = nn.BatchNorm2d(16)
        self.conv2d_5_rgb_attention = torch.nn.Sequential()
        self.conv2d_5_rgb_attention.add_module('conv2d_5_rgb_attention', nn.Conv2d(16, 8, kernel_size=3, padding=1))
        self.bn_rgb_5 = nn.BatchNorm2d(8)

        self.AdaptiveAverPool_8 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.AdaptiveAverPool_16 = nn.AvgPool2d(kernel_size=4, stride=4)

        self.upsample_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample_2 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.conv2d_1_rgb_red_concat_5 = nn.Conv2d(6, 3, kernel_size=3, padding=1)
        self.conv2d_1_rgb_red_concat_5_bn = nn.BatchNorm2d(3)
        self.conv2d_1_rgb_red_concat_10 = nn.Conv2d(6, 3, kernel_size=3, padding=1)
        self.conv2d_1_rgb_red_concat_10_bn = nn.BatchNorm2d(3)

        self.GatedConv2dWithActivation = MaskConv2d(in_channels=(3 * 2),
                                                                   out_channels=3, kernel_size=3,
                                                                   stride=1, padding=1, activation=None)

        self.bn_cat = nn.BatchNorm2d(16)

        self.sigmoid = torch.nn.Sequential()
        self.sigmoid.add_module('Sigmoid', nn.Sigmoid())

        self.epsilon = 0.0001

        self.conv1 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.bn_end1 = nn.BatchNorm2d(4)

        self.conv2 = nn.Conv2d(3, c2, kernel_size=3, padding=1)
        self.bn_end2 = nn.BatchNorm2d(c2)

    def forward(self, x):
        og_im = x

        x = Variable(og_im[:, 0].unsqueeze(1))

        weight_hori = self.weight_const_hori
        weight_horix = Variable(weight_hori)
        weight_vertical = self.weight_const_vertical
        weight_verticaly = Variable(weight_vertical)
        try:
            x_hori = F.conv2d(x, weight_horix, padding=1)
        except:
            print('horizon error')
        try:
            x_vertical = F.conv2d(x, weight_verticaly, padding=1)
        except:
            print('vertical error')
        e = self.epsilon

        edge_detect = torch.cat((x_hori, x_vertical), 1)

        edge_detect_conved = self.conv2d_1_attention(edge_detect)  # 1,8
        edge_detect_conved = self.bn_edge_1(edge_detect_conved)
        edge_detect_conved = self.silu(edge_detect_conved)
        edge_detect_conved = self.conv2d_2_attention(edge_detect_conved)  # 8,16
        edge_detect_conved = self.bn_edge_2(edge_detect_conved)
        edge_detect_conved = self.silu(edge_detect_conved)

        edge_detect_conved = self.conv2d_4_attention(edge_detect_conved)  # 16,8
        edge_detect_conved = self.bn_edge_4(edge_detect_conved)
        edge_detect_conved = self.silu(edge_detect_conved)
        edge_detect_conved = self.conv2d_5_attention(edge_detect_conved)  # 8,1

        sigmoid_output = self.sigmoid(edge_detect_conved)
        edge_detect_conved = self.gamma * (sigmoid_output * og_im) + (1 - self.gamma) * og_im
        # print(self.gamma)

        edge_detect_conved = self.conv2(edge_detect_conved)
        edge_detect_conved = self.bn_end2(edge_detect_conved)
        edge_detect_conved = self.silu(edge_detect_conved)

        return edge_detect_conved


class TIENet(nn.Module):
    def __init__(self, c1, c2):
        super(TIENet, self).__init__()
        kernel_const_hori = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype='float32')
        kernel_const_hori = torch.FloatTensor(kernel_const_hori).unsqueeze(0)
        self.weight_const_hori = nn.Parameter(data=kernel_const_hori, requires_grad=False)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.delta = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        kernel_const_vertical = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype='float32')
        kernel_const_vertical = torch.FloatTensor(kernel_const_vertical).unsqueeze(0)
        self.weight_const_vertical = nn.Parameter(data=kernel_const_vertical, requires_grad=False)
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()

        self.conv2d_1_attention = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn_edge_1 = nn.BatchNorm2d(8)
        self.conv2d_2_attention = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn_edge_2 = nn.BatchNorm2d(16)
        self.conv2d_3_attention = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn_edge_3 = nn.BatchNorm2d(16)
        self.conv2d_4_attention = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.bn_edge_4 = nn.BatchNorm2d(8)
        self.conv2d_5_attention = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        self.bn_edge_5 = nn.BatchNorm2d(1)

        self.conv2d_1_rgb_attention = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.bn_rgb_1 = nn.BatchNorm2d(8)
        self.conv2d_2_rgb_attention = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn_rgb_2 = nn.BatchNorm2d(16)
        self.conv2d_3_rgb_attention = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn_rgb_3 = nn.BatchNorm2d(32)
        self.conv2d_4_rgb_attention = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn_rgb_4 = nn.BatchNorm2d(16)
        self.conv2d_5_rgb_attention = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.bn_rgb_5 = nn.BatchNorm2d(8)

        self.AdaptiveAverPool_8 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.AdaptiveAverPool_16 = nn.AvgPool2d(kernel_size=4, stride=4)

        self.upsample_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample_2 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.conv2d_1_rgb_red_concat_5 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv2d_1_rgb_red_concat_5_bn = nn.BatchNorm2d(8)
        self.conv2d_1_rgb_red_concat_10 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv2d_1_rgb_red_concat_10_bn = nn.BatchNorm2d(8)

        self.GatedConv2dWithActivation = MaskConv2d(in_channels=(8 * 2),
                                                                   out_channels=8, kernel_size=3,
                                                                   stride=1, padding=1, activation=None)

        self.bn_cat = nn.BatchNorm2d(16)

        self.sigmoid = nn.Sigmoid()

        self.conv2d_1_1 = nn.Conv2d(9, 1, kernel_size=3, padding=1)
        self.bn_1_1 = nn.BatchNorm2d(1)

        self.epsilon = 0.0001

        self.conv1 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.bn_end1 = nn.BatchNorm2d(4)

        self.conv2 = nn.Conv2d(3, c2, kernel_size=3, padding=1)
        self.bn_end2 = nn.BatchNorm2d(c2)

    def forward(self, x):
        # global x_hori, x_vertical
        og_im = x
        # shape1 = og_im.shape[2]
        # shape2 = og_im.shape[3]

        x = Variable(og_im[:, 0].unsqueeze(1))
        y = Variable(og_im[:, 1].unsqueeze(1))
        z = Variable(og_im[:, 2].unsqueeze(1))
        x = (x + y + z) / 3
        # x = x.cuda()
        # print(x)

        # 反向传播可更新参数
        weight_hori = self.weight_const_hori
        weight_horix = Variable(weight_hori)
        weight_vertical = self.weight_const_vertical
        weight_verticaly = Variable(weight_vertical)

        try:
            x_hori = F.conv2d(x, weight_horix, padding=1)
        except:
            print('horizon error')
        try:
            x_vertical = F.conv2d(x, weight_verticaly, padding=1)
        except:
            print('vertical error')
        e = self.epsilon

        # get edge image 归一化
        # edge_detect = (torch.add(x_hori.pow(2), x_vertical.pow(2))).pow(0.5)
        edge_detect = (abs(x_hori) * 0.5 + abs(x_vertical) * 0.5)

        # edge_detect = torch.cat((x_hori, x_vertical), 1)
        # convolution of edge image
        edge_detect_conved = self.conv2d_1_attention(edge_detect)  # 1,8
        edge_detect_conved = self.bn_edge_1(edge_detect_conved)
        edge_detect_conved = self.silu(edge_detect_conved)
        edge_detect_conved = self.conv2d_2_attention(edge_detect_conved)  # 8,16
        edge_detect_conved = self.bn_edge_2(edge_detect_conved)
        edge_detect_conved = self.silu(edge_detect_conved)
        # edge_detect_conved = self.GatedConv2dWithActivation(edge_detect_conved)  # ch=8

        edge_detect_conved = self.conv2d_4_attention(edge_detect_conved)  # 16,8
        edge_detect_conved = self.bn_edge_4(edge_detect_conved)
        edge_detect_conved = self.silu(edge_detect_conved)
        edge_detect_conved = self.conv2d_5_attention(edge_detect_conved)  # 8,1
        edge_detect_conved = self.bn_edge_5(edge_detect_conved)
        edge_detect_conved = self.silu(edge_detect_conved)

        edge_detect = edge_detect/(edge_detect.max() + e)

        rgb_red = torch.cat((og_im, edge_detect), 1)  # 4

        rgb_conved = self.conv2d_1_rgb_attention(rgb_red)  # 4,8
        rgb_conved = self.bn_rgb_1(rgb_conved)
        rgb_conved = self.silu(rgb_conved)
        rgb_conved = self.conv2d_2_rgb_attention(rgb_conved)  # 8,16
        rgb_conved = self.bn_rgb_2(rgb_conved)
        rgb_conved = self.silu(rgb_conved)
        rgb_conved = self.conv2d_5_rgb_attention(rgb_conved)  # 16,8
        rgb_conved = self.bn_rgb_5(rgb_conved)
        rgb_conved = self.silu(rgb_conved)

        x_pooled_8 = self.AdaptiveAverPool_8(rgb_conved)  # 2
        x_pooled_16 = self.AdaptiveAverPool_8(x_pooled_8)  # 4
        # x_pooled_32 = self.AdaptiveAverPool_32(rgb_conved)  # 8

        x_pooled_upsample_8 = self.upsample_1(x_pooled_8)  # 2
        x_pooled_upsample_16 = self.upsample_2(x_pooled_16)  # 4

        x_concat_8 = torch.cat((rgb_conved, x_pooled_upsample_8), 1)  # ch=8+8
        x_concat_16 = torch.cat((rgb_conved, x_pooled_upsample_16), 1)

        x_concat_8_out = self.conv2d_1_rgb_red_concat_5(x_concat_8)  # 8
        x_concat_8_out = self.conv2d_1_rgb_red_concat_5_bn(x_concat_8_out)  # 8
        x_concat_8_out = self.silu(x_concat_8_out)
        x_concat_16_out = self.conv2d_1_rgb_red_concat_10(x_concat_16)
        x_concat_16_out = self.conv2d_1_rgb_red_concat_10_bn(x_concat_16_out)
        x_concat_16_out = self.silu(x_concat_16_out)

        x_gated_conv_input = torch.cat((x_concat_8_out, x_concat_16_out), 1)  # ch=16

        x_gated_conv_output = self.MaskConv2d(x_gated_conv_input)  # ch=8

        rgb_red_conved = torch.cat((x_gated_conv_output, edge_detect_conved), 1)
        rgb_red_conved = self.conv2d_1_1(rgb_red_conved)
        rgb_red_conved = self.bn_1_1(rgb_red_conved)
        rgb_red_conved = self.silu(rgb_red_conved)
        gamma = self.sigmoid(self.gamma)

        sigmoid_output = self.sigmoid(rgb_red_conved)
        edge_detect_conved = gamma * (sigmoid_output * rgb_red) + (1-gamma) * rgb_red
        print(gamma)

        return edge_detect_conved
