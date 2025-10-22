#!/usr/bin/python3
# coding=utf-8
import torch.nn as nn
import torch.nn.functional as F
import torch


from model.modules.weight_init import weight_init


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        # 使用1x1卷积来计算通道注意力
        self.fc1 = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // 16, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        # 通过两层全连接计算通道注意力
        attention = F.relu(self.fc1(avg_pool))
        attention = self.fc2(attention)
        attention = self.sigmoid(attention)
        return x * attention  # 对输入特征图进行加权
    def initialize(self):
        weight_init(self)
    
class Conv64to32(nn.Module):
    def __init__(self, in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1):
        super(Conv64to32, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)   # 卷积操作
        x = self.bn(x)     # 批量归一化
        x = self.relu(x)   # 激活函数
        return x
    def initialize(self):
        weight_init(self)
    

class MGCIA1(nn.Module):
    def __init__(self):
        super(MGCIA1, self).__init__()

        # for body
        self.conv1_b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_b = nn.BatchNorm2d(32)

        self.detail_a = AttentionModule(32)
        self.body_a = AttentionModule(32)
      
        self.conv2_b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2_b = nn.BatchNorm2d(32)

        # for detail
        self.conv1_d = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_d = nn.BatchNorm2d(32)

        self.conv2_d = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2_d = nn.BatchNorm2d(32)

        # after concat
        self.conv3_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(32)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(32)
        self.conv3_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm2d(32)

        self.fc = nn.Linear(in_features=64, out_features=32)  # 128 改为 64，输出通道改为 32

    def forward(self, body_map, detail_map):
        body_map_size = body_map.size()[2:]

        body_conv1 = self.body_a(body_map)
        detail_conv1 = self.detail_a(detail_map)

        AfterAdd_body   = body_map + detail_conv1
        AfterAdd_detail = body_conv1 + detail_map

        body_conv2   = F.relu(self.bn2_b(self.conv2_b(AfterAdd_body)), inplace=True)
        detail_conv2 = F.relu(self.bn2_d(self.conv2_d(AfterAdd_detail)), inplace=True)

        AfterAdd_BD = body_conv2 + detail_conv2

        AfterAVE = AfterAdd_BD.mean(-1).mean(-1)
        AfterMAX = AfterAdd_BD.max(-1)[0].max(-1)[0]

        AfterCat_MaxAve = torch.cat([AfterAVE, AfterMAX], dim=1)

        AfterFC = self.fc(AfterCat_MaxAve)
        AfterFC_unsqueeze = AfterFC.unsqueeze(-1).unsqueeze(-1)

        AfterMul_conv1 = F.relu(self.bn3_1(self.conv3_1(AfterFC_unsqueeze * AfterAdd_BD)), inplace=True)
        AfterMul_conv2 = F.relu(self.bn3_2(self.conv3_2(AfterMul_conv1)), inplace=True)
        AfterMul_conv3 = F.relu(self.bn3_3(self.conv3_3(AfterMul_conv2)), inplace=True)

        out = AfterMul_conv3

        return out
    def initialize(self):
        weight_init(self)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    def initialize(self):
        weight_init(self)

class MGCIB1(nn.Module):
    def __init__(self):
        super(MGCIB1, self).__init__()

        # for body
        self.conv1_b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_b = nn.BatchNorm2d(32)

        self.detail_a = AttentionModule(32)
        self.body_a = AttentionModule(32)
      
        self.conv2_b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2_b = nn.BatchNorm2d(32)

        # for detail
        self.conv1_d = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_d = nn.BatchNorm2d(32)

        self.conv2_d = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2_d = nn.BatchNorm2d(32)

        # after concat
        self.conv3_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(32)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(32)
        self.conv3_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm2d(32)

        self.fc = nn.Linear(in_features=64, out_features=32)  # 128 改为 64，输出通道改为 32
        self.sa = SpatialAttention()

        self.conva = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    def forward(self, body_map, detail_map):
        body_map_size = body_map.size()[2:]
    
        body_conv1 = self.body_a(body_map)
        detail_conv1 = self.detail_a(detail_map)

        AfterAdd_body   = body_map + detail_conv1
        AfterAdd_detail = body_conv1 + detail_map

        body_conv2   = F.relu(self.bn2_b(self.conv2_b(AfterAdd_body)), inplace=True)
        detail_conv2 = F.relu(self.bn2_d(self.conv2_d(AfterAdd_detail)), inplace=True)

        AfterAdd_BD = body_conv2 + detail_conv2

        AfterAVE = AfterAdd_BD.mean(-1).mean(-1)
        AfterMAX = AfterAdd_BD.max(-1)[0].max(-1)[0]

        AfterCat_MaxAve = torch.cat([AfterAVE, AfterMAX], dim=1)

        AfterFC = self.fc(AfterCat_MaxAve)
        AfterFC_unsqueeze = AfterFC.unsqueeze(-1).unsqueeze(-1)

        AfterMul_conv1 = F.relu(self.bn3_1(self.conv3_1(AfterFC_unsqueeze * AfterAdd_BD)), inplace=True)
        AfterMul_conv2 = F.relu(self.bn3_2(self.conv3_2(AfterMul_conv1)), inplace=True)
        AfterMul_conv3 = F.relu(self.bn3_3(self.conv3_3(AfterMul_conv2)), inplace=True)

        out = AfterMul_conv3
        return out
    def initialize(self):
        weight_init(self)
      


class MGCIB2(nn.Module):
    def __init__(self):
        super(MGCIB2, self).__init__()

        # for body
        self.conv1_b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_b = nn.BatchNorm2d(32)

        self.detail_a = AttentionModule(32)
        self.body_a = AttentionModule(32)
      
        self.conv2_b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2_b = nn.BatchNorm2d(32)

        # for detail
        self.conv1_d = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_d = nn.BatchNorm2d(32)

        self.conv2_d = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2_d = nn.BatchNorm2d(32)

        # after concat
        self.conv3_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(32)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(32)
        self.conv3_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm2d(32)

        self.fc = nn.Linear(in_features=64, out_features=32)  # 128 改为 64，输出通道改为 32
        self.sa = SpatialAttention()
        self.convb = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    def forward(self, body_map, detail_map):
        body_map_size = body_map.size()[2:]
        body_conv1 = self.body_a(body_map)
        detail_conv1 = self.detail_a(detail_map)

        AfterAdd_body   = body_map + detail_conv1
        AfterAdd_detail = body_conv1 + detail_map

        body_conv2   = F.relu(self.bn2_b(self.conv2_b(AfterAdd_body)), inplace=True)
        detail_conv2 = F.relu(self.bn2_d(self.conv2_d(AfterAdd_detail)), inplace=True)

        AfterAdd_BD = body_conv2 + detail_conv2
        x = self.sa(AfterAdd_BD)
        m = x*AfterAdd_BD
        out = self.convb(m)
        return out
    def initialize(self):
        weight_init(self)


class MGCIB3(nn.Module):
    def __init__(self):
        super(MGCIB3, self).__init__()

        # for body
        self.conv1_b = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn1_b = nn.BatchNorm2d(16)

        self.detail_a = AttentionModule(16)
        self.body_a = AttentionModule(16)
      
        self.conv2_b = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn2_b = nn.BatchNorm2d(16)

        # for detail
        self.conv1_d = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn1_d = nn.BatchNorm2d(16)

        self.conv2_d = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn2_d = nn.BatchNorm2d(16)
        self.conv111 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        # after concat
        self.conv3_1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(16)
        self.conv3_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(16)
        self.conv3_3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm2d(16)

        self.fc = nn.Linear(in_features=32, out_features=16)  # 128 改为 64，输出通道改为 32
        self.sa = SpatialAttention()
        self.convc = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, body_map, detail_map):
        body_map_size = body_map.size()[2:]
        body_map = self.conv111(body_map)
        body_conv1 = self.body_a(body_map)
        detail_conv1 = self.detail_a(detail_map)
        AfterAdd_body   = body_map + detail_conv1
        AfterAdd_detail = body_conv1 + detail_map

        body_conv2   = F.relu(self.bn2_b(self.conv2_b(AfterAdd_body)), inplace=True)
        detail_conv2 = F.relu(self.bn2_d(self.conv2_d(AfterAdd_detail)), inplace=True)

        AfterAdd_BD = body_conv2 + detail_conv2
        x = self.sa(AfterAdd_BD)
        m = x*AfterAdd_BD
        out = self.convc(m)

        return out
    def initialize(self):
        weight_init(self)





class MGCIDe(nn.Module):
    def __init__(self):
        super(MGCIDe, self).__init__()
        self.MGCIA1_0 = MGCIA1()
        self.MGCIA1_1 = MGCIA1()
        self.MGCIA1_2 = MGCIA1()
        self.MGCIB_1 = MGCIB1()
        self.MGCIB_2 = MGCIB2()
        self.MGCIB_3 = MGCIB3()

        self.conv_1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv_4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv_4_reduce = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1),
                                           nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.conv_5 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(16), nn.ReLU(inplace=True))

        self.conv1 = Conv64to32()
    def forward(self, feat_trunk, feat_struct):
        feat_trunk[0] = self.conv1(feat_trunk[0])
        mask = self.MGCIA1_0(feat_trunk[0], feat_struct[0])
        feat_trunk[1] = self.conv1(feat_trunk[1])
        temp = self.MGCIA1_1(feat_trunk[1], feat_struct[1])

        maskup = F.interpolate(mask, size=temp.size()[2:], mode='bilinear')
        temp = maskup + temp
        mask = self.conv_1(temp)
        feat_trunk[2] = self.conv1(feat_trunk[2])

        temp = self.MGCIA1_2(feat_trunk[2], feat_struct[2])
        maskup = F.interpolate(mask, size=temp.size()[2:], mode='bilinear')
        temp = maskup + temp
        mask = self.conv_2(temp)

        maskup = F.interpolate(mask, size=feat_struct[3].size()[2:], mode='bilinear')
        temp = self.MGCIB_1(maskup, feat_struct[3])
        temp = maskup + temp
        mask = self.conv_3(temp)

        maskup = F.interpolate(mask, size=feat_struct[4].size()[2:], mode='bilinear')
        temp = self.MGCIB_2(maskup, feat_struct[4])
        temp = maskup + temp
        mask = self.conv_4(temp)

        maskup = F.interpolate(mask, size=feat_struct[5].size()[2:], mode='bilinear')
        temp = self.MGCIB_3(maskup, feat_struct[5])
        maskup = self.conv_4_reduce(maskup)
        temp = maskup + temp
        mask = self.conv_5(temp)

        return mask

    def initialize(self):
        weight_init(self)