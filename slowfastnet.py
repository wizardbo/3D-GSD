import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch as tensor
from torch.autograd import Variable
from functools import reduce

__all__ = ['resnet50', 'resnet101','resnet152', 'resnet200']

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

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.se = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output



class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual

# if __name__ == "__main__":
#     input = torch.randn(50, 512, 4,  7, 7)
#     cbam = CBAMBlock(channel=512, reduction=16)
#     output = cbam(input)
#     print(output.shape)
#     exit()

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1):
        super(Bottleneck, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=(1,stride,stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class SlowFast(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], class_num=10, dropout=0.5 ):
        super(SlowFast, self).__init__()

        self.fast_inplanes = 8
        self.fast_conv1 = nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast_bn1 = nn.BatchNorm3d(8)
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fast_res2 = self._make_layer_fast(block, 8, layers[0], head_conv=3)
        self.fast_res3 = self._make_layer_fast(
            block, 16, layers[1], stride=2, head_conv=3)
        self.fast_res4 = self._make_layer_fast(
            block, 32, layers[2], stride=2, head_conv=3)
        self.fast_res5 = self._make_layer_fast(
            block, 64, layers[3], stride=2, head_conv=3)
        
        self.lateral_p1 = nn.Conv3d(8, 8*2, kernel_size=(5, 1, 1), stride=(8, 1 ,1), bias=False, padding=(2, 0, 0))
        self.lateral_res2 = nn.Conv3d(32,32*2, kernel_size=(5, 1, 1), stride=(8, 1 ,1), bias=False, padding=(2, 0, 0))
        self.lateral_res3 = nn.Conv3d(64,64*2, kernel_size=(5, 1, 1), stride=(8, 1 ,1), bias=False, padding=(2, 0, 0))
        self.lateral_res4 = nn.Conv3d(128,128*2, kernel_size=(5, 1, 1), stride=(8, 1 ,1), bias=False, padding=(2, 0, 0))

        self.slow_inplanes = 64+64//8*2
        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        # sknet
        # self.conv = SKConv(32, 1,2,16,32)
        # self.skcat = nn.Conv3d(240, 80, kernel_size=(1, 1, 1))  #[1, 80, 4, 56, 56]
        # self.slow_SK3D = SKLayer_3D(channel=80, reduction=4)
        # self.skcat = nn.Conv3d(240, 80, kernel_size=(1, 1, 1))
        self.slow_res2 = self._make_layer_slow(block, 64, layers[0], head_conv=1)
        self.slow_res3 = self._make_layer_slow(
            block, 128, layers[1], stride=2, head_conv=1)
        self.slow_res4 = self._make_layer_slow(
            block, 256, layers[2], stride=2, head_conv=3)
        self.slow_res5 = self._make_layer_slow(
            block, 512, layers[3], stride=2, head_conv=3)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(self.fast_inplanes+2048, class_num, bias=False)
        self.scbam1 = CBAMBlock(channel=64, reduction=16)
        self.scbam2 = CBAMBlock(channel=64, reduction=16)
        self.scbam3 = CBAMBlock(channel=64, reduction=16)
        #self.scat   = nn.Conv3d(6144, 2048, kernel_size=(4, 1, 1)) # [1,2048, 4, 7, 7]
        self.scat = nn.Conv3d(192, 64, kernel_size=(1, 1, 1))  # debug 1:  torch.Size([1, 64, 4, 112, 112])

        # self.fcbam1 = CBAMBlock(channel=256, reduction=16)
        # self.fcbam2 = CBAMBlock(channel=256, reduction=16)
        # self.fcbam3 = CBAMBlock(channel=256, reduction=16)
        self.fcbam1 = CBAMBlock(channel=8, reduction=2)
        self.fcbam2 = CBAMBlock(channel=8, reduction=2)
        self.fcbam3 = CBAMBlock(channel=8, reduction=2)
        # self.fcat = nn.Conv3d(768, 256, kernel_size=(4, 1, 1)) # [1,256, 32, 7, 7]
        self.fcat = nn.Conv3d(24, 8, kernel_size=(1, 1, 1))  # debug 2:  torch.Size([1, 8, 32, 112, 112])

    def forward(self, input):
        fast, lateral = self.FastPath(input[:, :, ::2, :, :])
        slow = self.SlowPath(input[:, :, ::16, :, :], lateral)

        x = torch.cat([slow, fast], dim=1)
        x = self.dp(x)
        x = self.fc(x)
        return x

    def SlowPath(self, input, lateral):
        x = self.slow_conv1(input)
        #print('debug 1: ', x.size())  # debug 1:  torch.Size([1, 64, 4, 112, 112])
        x_1 = self.scbam1(x)
        x_2 = self.scbam2(x)
        x_3 = self.scbam3(x)
        x = torch.cat([x_1, x_2, x_3], dim=1)
        x = self.scat(x)
        # print('debug 2: ', x.size())  # debug 1:  torch.Size([1, 64, 1, 112, 112])

        x = self.slow_bn1(x)
        x = self.slow_relu(x)
        x = self.slow_maxpool(x)
        x = torch.cat([x, lateral[0]],dim=1)

        # print('debug 2: ', x.size())  # debug 1:  torch.Size([1, 80, 4, 56, 56])
        # sknet
        # x = self.conv(x)
        # x = torch.cat([x], dim=1)  # x = torch.cat([x, lateral[0]], dim=1)
        # x = self.skcat(x)
        #x = self.slow_SK3D(x)

        x = self.slow_res2(x)
        x = torch.cat([x, lateral[1]],dim=1)
        x = self.slow_res3(x)
        x = torch.cat([x, lateral[2]],dim=1)
        x = self.slow_res4(x)
        x = torch.cat([x, lateral[3]],dim=1)
        x = self.slow_res5(x)
        # print('debug 1: ', x.size())  # [1,2048, 4, 7, 7]
        # x_1 = self.scbam1(x)
        # x_2 = self.scbam2(x)
        # x_3 = self.scbam3(x)
        # x = torch.cat([x_1, x_2, x_3], dim=1)
        # x = self.scat(x)
        # print('debug 1: ', x.size())  # [1,2048, 4, 7, 7]
        x = nn.AdaptiveAvgPool3d(1)(x)
        # print('fast', fast.size()) #[1,256]
        x = x.view(-1, x.size(1))
        return x

    def FastPath(self, input):
        lateral = []
        x = self.fast_conv1(input)
        # print('debug 1: ', x.size()) # debug 2:  torch.Size([1, 8, 32, 112, 112])

        x_1 = self.fcbam1(x)
        x_2 = self.fcbam2(x)
        x_3 = self.fcbam3(x)
        x = torch.cat([x_1, x_2, x_3], dim=1)
        x = self.fcat(x)
        # print('debug 2: ', x.size()) # debug 2:  torch.Size([1, 8, 32, 112, 112])

        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        pool1 = self.fast_maxpool(x)
        lateral_p = self.lateral_p1(pool1)
        lateral.append(lateral_p)
        # print('debug 3: ', x.size()) # debug 2:  torch.Size([1, 8, 32, 112, 112])

        res2 = self.fast_res2(pool1)
        lateral_res2 = self.lateral_res2(res2)
        lateral.append(lateral_res2)

        
        res3 = self.fast_res3(res2)
        lateral_res3 = self.lateral_res3(res3)
        lateral.append(lateral_res3)

        res4 = self.fast_res4(res3)
        lateral_res4 = self.lateral_res4(res4)
        lateral.append(lateral_res4)

        res5 = self.fast_res5(res4)
        # x = self.fast_res5(res4)
        # print('debug 2: ', x.size())  # [1,256, 32, 7, 7]
        # x_1 = self.fcbam1(x)
        # x_2 = self.fcbam2(x)
        # x_3 = self.fcbam3(x)
        # x = torch.cat([x_1, x_2, x_3], dim=1)
        # x = self.fcat(x)
        # print('debug 2: ', x.size())  # [1,256, 32, 7, 7]
        # x = nn.AdaptiveAvgPool3d(1)(x)
        x = nn.AdaptiveAvgPool3d(1)(res5)
        x = x.view(-1, x.size(1))
        return x, lateral

    def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.fast_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.fast_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1,stride,stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.fast_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.fast_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast_inplanes, planes, head_conv=head_conv))
        return nn.Sequential(*layers)

    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.slow_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.slow_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1,stride,stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.slow_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.slow_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.slow_inplanes, planes, head_conv=head_conv))
  
        self.slow_inplanes = planes * block.expansion + planes * block.expansion//8*2
        return nn.Sequential(*layers)




def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = SlowFast(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

if __name__ == "__main__":
    num_classes = 157 #101 #157
    input_tensor = torch.autograd.Variable(torch.rand(4, 3, 64, 224, 224))
    model = resnet50(class_num=num_classes)
    output = model(input_tensor)
    print(output.size())

    # input = torch.randn(50, 512, 7, 7)
    # kernel_size = input.shape[2]
    # cbam = CBAMBlock(channel=512, reduction=16, kernel_size=kernel_size)
    # output = cbam(input)
    # print(output.shape)






