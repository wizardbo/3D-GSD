import sys
sys.path.insert(0, '')

import math
import numpy as np
import torch
import torchvision
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.models as models

from utils import import_class, count_params
from model.ms_gcn import MultiScale_GraphConv as MS_GCN
from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from model.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from model.mlp import MLP
from model.activation import activation_factory
from functools import reduce
from torchvision.models.video import r3d_18

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class PoseEstimationModel(nn.Module):
    def __init__(self):
        super(PoseEstimationModel, self).__init__()
        self.conv1 = nn.Conv2d(384, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(128, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)
        self.layer4 = self._make_layer(512, 1024, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 60) # 60是NTU 60数据集的动作种类数

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for i in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class DensePoseEstimation(nn.Module):
    def __init__(self, input_channels=384, output_channels=384):
        super(DensePoseEstimation, self).__init__()

        # 降维
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1)

        # 升维
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=1)
        self.upconv2 = nn.ConvTranspose2d(64, 256, kernel_size=1)
        self.upconv3 = nn.ConvTranspose2d(256, output_channels, kernel_size=1)

    def forward(self, x):
        # 降维
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 升维
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = self.upconv3(x)

        return x

# class PoseEstimation(nn.Module):
#     def __init__(self, num_joints):
#         '''其中，num_joints是输出中关键点的数量。'''
#         super(PoseEstimation, self).__init__()
#
#         # 加载预训练的ResNet模型
#         self.resnet = models.resnet50(pretrained=True)
#
#         # 修改ResNet的最后一层，使其输出num_joints * 2个通道
#         num_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_features, num_joints * 2)
#
#         # 定义输出矩阵的形状
#         self.num_joints = num_joints
#         self.output_shape = (-1, self.num_joints,12, 2)
#
#     def forward(self, x):
#         x = self.resnet(x)
#         x = x.reshape(self.output_shape)
#         return x

class SKLayer_3D(nn.Module):
    def __init__(self, channel, M=2, reduction=4, L=4, G=4):
        '''

        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param M:  分支数
        :param reduction: 降维时的缩小比例
        :param L:  降维时全连接层 神经元的下界
        :param G:  组卷积
        '''
        super(SKLayer_3D, self).__init__()

        self.M = M
        self.channel = channel

        # 尺度不变
        self.conv = nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(self.M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(nn.Sequential(nn.Conv3d(channel,
                                                     channel,
                                                     3,
                                                     1,
                                                     padding=1 + i,
                                                     dilation=1 + i,
                                                     groups=G,
                                                     bias=False),
                                           nn.BatchNorm3d(channel),
                                           nn.ReLU(inplace=True)))
        self.fbap = nn.AdaptiveAvgPool3d(1)  # 三维自适应pool到指定维度    这里指定为1，实现 三维GAP
        d = max(channel // reduction, L)  # 计算向量Z 的长度d   下限为L
        self.fc1 = nn.Sequential(nn.Conv3d(in_channels=channel, out_channels=d, kernel_size=(1, 1, 1), bias=False),
                                 nn.BatchNorm3d(d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv3d(in_channels=d, out_channels=channel * M, kernel_size=(1, 1, 1), bias=False)  # 升维
        self.softmax = nn.Softmax(dim=1)  # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1

    def forward(self, input):
        #batch_size, channel, _, _, _ = input.shape
        batch_size, channel,_,  _, _ = input.size()

        # split阶段
        output = []
        for i, conv in enumerate(self.conv):
            output.append(conv(input))

        # fusion阶段
        U = output[0] + output[1]  # 逐元素相加生成 混合特征U
        s = self.fbap(U)
        z = self.fc1(s)  # S->Z降维
        a_b = self.fc2(z)  # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        a_b = a_b.reshape(batch_size, self.M, channel, 1, 1, 1)  # 调整形状，变为 两个全连接层的值
        a_b = self.softmax(a_b)  # 使得两个全连接层对应位置进行softmax

        # selection阶段
        a_b = list(a_b.chunk(self.M, dim=1))  # split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a_b = list(map(lambda x: t.squeeze(x,dim=1), a_b))  # 压缩第一维
        V = list(map(lambda x, y: x * y, output, a_b))  # 权重与对应  不同卷积核输出的U 逐元素相乘
        V = V[0] + V[1]  # 两个加权后的特征 逐元素相加
        return V


class MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 window_stride,
                 window_dilation,
                 embed_factor=1,
                 activation='relu'):
        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        # self.openpose = OpenPoseEstimation(25)
        # self.openpose = OpenPoseEstimationMSG3D2()
        # self.openpose = PoseEstimation1(25)
        # self.openpose = DeepPose(25)
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            # The first STGC block changes channels right away; others change at collapse
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = nn.Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            SpatialTemporal_MS_GCN(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A_binary=A_binary,
                num_scales=num_scales,
                window_size=window_size,
                use_Ares=True
            )
        )

        # self.sknet2D = SKNet(out_channels)
        # self.sknet3D= SKLayer_3D(out_channels, reduction=4)
        self.sknet3D = SKLayer_3D(self.embed_channels_out, reduction=4)
        #self.mgcat = nn.Conv3d(288, 96,kernel_size=(4, 1, 1))
        self.out_conv = nn.Conv3d(self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, _, T, V = x.shape
        # print('x ', x.size())
        # x = self.openpose(x)
        # print('openpose: ', x.size()) # debug 3:  torch.Size([12, 3, 50, 25])
        # x = self.openpose(x)

        x = self.in1x1(x)
        # print('debug in1x1: ', x.size())  # debug in1x1:  torch.Size([12, 3, 50, 25])
        # Construct temporal windows and apply MS-GCN

        x = self.gcn3d(x)
        # x = self.openpose(x)
        # print('debug gcn3d: ', x.size())  # debug gcn3d:  torch.Size([12, 96, 50, 75])
        # Collapse the window dimension
        x = x.view(N, self.embed_channels_out, -1, self.window_size, V)
        # print('debug view: ', x.size())  # debug view:  torch.Size([12, 96, 50, 3, 25])
        # x = self.sknet2D(x)
        #x = self.sknet3D(x)  # 添加修改
        # print('sknet3D: ', x.size())
        # x = torch.cat([x], dim=1)
        # x = self.mgcat(x)

        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)
        # print('out_bn: ', x.size())

        # no activation
        return x


class MultiWindow_MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_sizes=[3,5],
                 window_stride=1,
                 window_dilations=[1,1]):

        super().__init__()
        self.gcn3d = nn.ModuleList([
            MS_G3D(
                in_channels,
                out_channels,
                A_binary,
                num_scales,
                window_size,
                window_stride,
                window_dilation
            )
            for window_size, window_dilation in zip(window_sizes, window_dilations)
        ])

    def forward(self, x):
        # Input shape: (N, C, T, V)
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d(x)
        # no activation
        return out_sum


class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3):
        super(Model, self).__init__()

        Graph = import_class(graph)
        A_binary = Graph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.Densepose = DensePoseEstimation()

        # channels
        c1 = 96
        c2 = c1 * 2     # 192
        c3 = c2 * 2     # 384

        # r=3 STGC blocks
        self.gcn3d1 = MultiWindow_MS_G3D(3, c1, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1)

        self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2)

        self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn3 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = MS_TCN(c3, c3)

        self.fc = nn.Linear(c3, num_class)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()
        # print('sknet3D: ', x.size())
        # Apply activation to the sum of the pathways
        x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        x = self.tcn1(x)
        # print('tcn1: ', x.size())
        x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        x = self.tcn2(x)

        x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        x = self.tcn3(x)
        # print('tcn3: ', x.size())  # torch.Size([12, 384, 13, 25]) x = self.Densepose(x)   # 添加修改
        # x = self.Densepose(x)   # 添加修改
        out = x
        # print('N, C, T, V, M: ', N, C, T, V, M)
        # print('out: ', x.size())  # out:  torch.Size([12, 384, 13, 25])
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)   # Global Average Pooling (Spatial+Temporal)
        out = out.mean(1)   # Average pool number of bodies in the sequence

        out = self.fc(out)
        return out


if __name__ == "__main__":
    # For debugging purposes
    import sys
    sys.path.append('..')

    model = Model(
        num_class=60,
        num_point=25,
        num_person=2,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph='graph.ntu_rgb_d.AdjMatrixGraph'
    )

    N, C, T, V, M = 6, 3, 50, 25, 2  #N：batch size，C：通道数，T：时间步数，V：关节点数，M:骨架数据中包含的骨架数;
    x = torch.randn(N,C,T,V,M)
    model.forward(x)

    print('Model total # params:', count_params(model))
