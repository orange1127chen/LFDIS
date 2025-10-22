#!/usr/bin/python3
# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules.weight_init import weight_init




class BodyDe(nn.Module):
    def __init__(self):
        super(BodyDe, self).__init__()

        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        # self.mpca = MPCA()

    def forward(self, in_feat):
        
        out0 = F.relu(self.bn0(self.conv0(in_feat[0])), inplace=True)
        out0_up = F.interpolate(out0, size=in_feat[1].size()[2:], mode='bilinear')
        out1 = F.relu(self.bn1(self.conv1(out0_up + in_feat[1])), inplace=True)
        out1_up = F.interpolate(out1, size=in_feat[2].size()[2:], mode='bilinear')
        out2 = F.relu(self.bn2(self.conv2(out1_up + in_feat[2])), inplace=True)
        out2_up = F.interpolate(out2, size=in_feat[3].size()[2:], mode='bilinear')
        out3 = F.relu(self.bn3(self.conv3(out2_up + in_feat[3])), inplace=True)
        out3_up = F.interpolate(out3, size=in_feat[4].size()[2:], mode='bilinear')
        out4 = F.relu(self.bn4(self.conv4(out3_up + in_feat[4])), inplace=True)
    
        return out2, out3, out4
    def initialize(self):
        weight_init(self)

