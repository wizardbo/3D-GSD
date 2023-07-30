import sys
sys.path.insert(0, '')
import torchvision
import torch.nn as nn
import os



from torchvision.models.video import r3d_18

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# class DeepPose(nn.Module):
# 	"""docstring for DeepPose"""
# 	def __init__(self, nJoints, modelName='resnet50', T=50, V=25):
# 		super(DeepPose, self).__init__()
# 		self.nJoints = nJoints
# 		self.block = 'BottleNeck' if (int(modelName[6:]) > 34) else 'BasicBlock'
# 		self.resnet = getattr(torchvision.models, modelName)(pretrained=True)
# 		self.resnet.fc = nn.Linear(512 * (4 if self.block == 'BottleNeck' else 1), self.nJoints * 2)
# 	def forward(self, x):
# 		return self.resnet(x)

class DeepPose(nn.Module):

	"""docstring for DeepPose"""
	def __init__(self, nJoints, modelName='resnet50'):
		super(DeepPose, self).__init__()
		self.nJoints = nJoints
		self.block = 'BottleNeck' if (int(modelName[6:]) > 34) else 'BasicBlock'
		self.resnet = getattr(torchvision.models, modelName)(pretrained=True)
		self.resnet.fc = nn.Linear(512 * (4 if self.block == 'BottleNeck' else 1), self.nJoints * 2)
	def forward(self, x):
		return self.resnet(x)



class PoseEstimation(nn.Module):
    # def __init__(self, in_channels, num_classes):
    def __init__(self, num_classes):
        super(PoseEstimation, self).__init__()
        self.in_channels = 12
        self.num_classes = num_classes
        # self.conv1 = nn.Conv3d(channels, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv1 = nn.Conv3d(self.in_channels, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.channel = channel
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        print('debug 5: ', x.size())
        x = self.conv1(x)
        print('debug 4: ', x.size())
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class OpenPoseEstimation(nn.Module):
    def __init__(self, num_classes=2):
        super(OpenPoseEstimation, self).__init__()
        self.num_classes = num_classes
        self.backbone = r3d_18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.num_classes)
        self.conv1 = nn.Conv3d(12, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1)) #96.64
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        print('relu: ', x.size()) # torch.Size([3, 3, 50, 25])
        x = x.unsqueeze(3)
        x = self.backbone(x)
        print('backbone: ', x.size())
        x = self.avgpool(x)
        print('avgpool: ', x.size())
        x = x.view(x.size(0), -1)
        return x

class OpenPoseEstimationMSG3D(nn.Module):
    def __init__(self, num_classes=2):
        super(OpenPoseEstimationMSG3D, self).__init__()
        self.num_classes = num_classes
        self.embed_channels_out = 12
        self.openpose = OpenPoseEstimation(self.embed_channels_out)
        self.conv1 = nn.Conv3d(self.embed_channels_out, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.openpose(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


import torch.nn as nn
import torch.nn.functional as F


class DeepPose(nn.Module):
    def __init__(self):
        super(DeepPose, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 24)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        x = x.view(-1, 24, 2)
        return x

'''其中，输入的张量维度是4D，形状为(batch_size, 3, 227, 227)，输出的张量维度也是4D，形状为(batch_size, 24,
                                                           2)。DeepPose算法基于卷积神经网络，可以进行2D姿态预测。在MS - G3D网络中加入该算法，您只需要在MS - G3D网络中调用DeepPose()
即可。'''

# class SKConv(nn.Module):
#     def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
#         super(SKConv,self).__init__()
#         d=max(in_channels//r,L)
#         self.M=M
#         self.out_channels=out_channels
#         self.conv=nn.ModuleList()
#         for i in range(M):
#             self.conv.append(nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=32,bias=False),
#                                            nn.BatchNorm2d(out_channels),
#                                            nn.ReLU(inplace=True)))
#         self.global_pool=nn.AdaptiveAvgPool2d(1)
#         self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
#                                nn.BatchNorm2d(d),
#                                nn.ReLU(inplace=True))
#         self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)
#         self.softmax=nn.Softmax(dim=1)
#     def forward(self, input):
#         batch_size=input.size(0)
#         output=[]
#         #the part of split
#         for i,conv in enumerate(self.conv):
#             #print(i,conv(input).size())
#             output.append(conv(input))
#         #the part of fusion
#         U=reduce(lambda x,y:x+y,output)
#         s=self.global_pool(U)
#         z=self.fc1(s)
#         a_b=self.fc2(z)
#         a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1)
#         a_b=self.softmax(a_b)
#         #the part of selection
#         a_b=list(a_b.chunk(self.M,dim=1))#split to a and b
#         a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b))
#         V=list(map(lambda x,y:x*y,output,a_b))
#         V=reduce(lambda x,y:x+y,V)
#         return V
#
#
# class SKBlock(nn.Module):
#     expansion=2
#     def __init__(self,inplanes,planes,stride=1,downsample=None):
#         super(SKBlock,self).__init__()
#         self.conv1=nn.Sequential(nn.Conv2d(inplanes,planes,1,1,0,bias=False),
#                                  nn.BatchNorm2d(planes),
#                                  nn.ReLU(inplace=True))
#         self.conv2=SKConv(planes,planes,stride)
#         self.conv3=nn.Sequential(nn.Conv2d(planes,planes*self.expansion,1,1,0,bias=False),
#                                  nn.BatchNorm2d(planes*self.expansion))
#         self.relu=nn.ReLU(inplace=True)
#         self.downsample=downsample
#     def forward(self, input):
#         shortcut=input
#         output=self.conv1(input)
#         output=self.conv2(output)
#         output=self.conv3(output)
#         if self.downsample is not None:
#             shortcut=self.downsample(input)
#         output+=shortcut
#         return self.relu(output)
#
#
# class SKNet(nn.Module):
#     def __init__(self,nums_class=1000,block=SKBlock,nums_block_list=[3, 4, 6, 3]):
#         super(SKNet,self).__init__()
#         self.inplanes=64
#         self.conv=nn.Sequential(nn.Conv2d(3,64,7,2,3,bias=False),
#                                 nn.BatchNorm2d(64),
#                                 nn.ReLU(inplace=True))
#         self.maxpool=nn.MaxPool2d(3,2,1)
#         self.layer1=self._make_layer(block,128,nums_block_list[0],stride=1)
#         self.layer2=self._make_layer(block,256,nums_block_list[1],stride=2)
#         self.layer3=self._make_layer(block,512,nums_block_list[2],stride=2)
#         self.layer4=self._make_layer(block,1024,nums_block_list[3],stride=2)
#         self.avgpool=nn.AdaptiveAvgPool2d(1)
#         self.fc=nn.Linear(1024*block.expansion,nums_class)
#         self.softmax=nn.Softmax(-1)
#     def forward(self, input):
#         output=self.conv(input)
#         output=self.maxpool(output)
#         output=self.layer1(output)
#         output=self.layer2(output)
#         output=self.layer3(output)
#         output=self.layer4(output)
#         output=self.avgpool(output)
#         output=output.squeeze(-1).squeeze(-1)
#         output=self.fc(output)
#         output=self.softmax(output)
#         return output
#     def _make_layer(self,block,planes,nums_block,stride=1):
#         downsample=None
#         if stride!=1 or self.inplanes!=planes*block.expansion:
#             downsample=nn.Sequential(nn.Conv2d(self.inplanes,planes*block.expansion,1,stride,bias=False),
#                                      nn.BatchNorm2d(planes*block.expansion))
#         layers=[]
#         layers.append(block(self.inplanes,planes,stride,downsample))
#         self.inplanes=planes*block.expansion
#         for _ in range(1,nums_block):
#             layers.append(block(self.inplanes,planes))
#         return nn.Sequential(*layers)




class DeepPose(nn.Module):
	"""docstring for DeepPose"""
	def __init__(self, nJoints, modelName='resnet50'):
		super(DeepPose, self).__init__()
		self.nJoints = nJoints
		self.block = 'BottleNeck' if (int(modelName[6:]) > 34) else 'BasicBlock'
		self.resnet = getattr(torchvision.models, modelName)(pretrained=True)
		self.resnet.fc = nn.Linear(512 * (4 if self.block == 'BottleNeck' else 1), self.nJoints * 2)
	def forward(self, x):
		return self.resnet(x)


'''nJoints是一个整数，表示人体骨架中关节点的数量。在forward方法中，self.resnet(x)的输出是一个2nJoints维度的向量，
其中nJoints表示人体骨架中关节点的数量，2表示每个关节点的坐标值(x,y)，即返回的是一个2D的向量。'''

class PoseEstimation(nn.Module):
    def __init__(self, num_joints):
        super(PoseEstimation, self).__init__()

        # 加载预训练的ResNet模型
        self.resnet = models.resnet50(pretrained=True)

        # 修改ResNet的最后一层，使其输出num_joints * 2个通道
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_joints * 2)

        # 定义输出矩阵的形状
        self.num_joints = num_joints
        self.output_shape = (-1, self.num_joints, 2)

    def forward(self, x):
        x = self.resnet(x)
        x = x.reshape(self.output_shape)
        return x
    '''其中，num_joints是输出中关键点的数量。'''


import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

class OpenPoseEstimator(nn.Module):
    def __init__(self, nJoints, modelName='resnet50'):
        super(OpenPoseEstimator, self).__init__()
        self.nJoints = nJoints
        self.resnet = getattr(models, modelName)(pretrained=True)
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, self.nJoints*2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)

        heatmaps = x[:, :(self.nJoints*2):2]
        pafs = x[:, (self.nJoints*2):]

        heatmaps = self.sigmoid(heatmaps)

        return heatmaps, pafs


# class OpenPoseEstimation(nn.Module):
#     def __init__(self, num_joints):
#         super(OpenPoseEstimation, self).__init__()
#
#         # input shape: [batch_size, num_channels=3, height=50, width=50]
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(num_features=32)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2)
#
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(num_features=64)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
#
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(num_features=128)
#         self.relu3 = nn.ReLU()
#         self.pool3 = nn.MaxPool2d(kernel_size=2)
#
#         # output shape: [batch_size, 128, height=13, width=13]
#         self.fc1 = nn.Linear(in_features=128 * 13 * 13, out_features=512)
#         self.relu4 = nn.ReLU()
#         self.dropout1 = nn.Dropout(p=0.5)
#
#         self.fc2 = nn.Linear(in_features=512, out_features=num_joints * 2)
#
#     def forward(self, x):
#         x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
#         x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
#         x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
#         x = x.view(x.size(0), -1)
#         x = self.dropout1(self.relu4(self.fc1(x)))
#         x = self.fc2(x)
#         return x

class OpenPoseEstimator(nn.Module):
    def __init__(self, nJoints, modelName='resnet50'):
        super(OpenPoseEstimator, self).__init__()
        self.nJoints = nJoints
        self.resnet = getattr(models, modelName)(pretrained=True)
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, self.nJoints*2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)

        heatmaps = x[:, :(self.nJoints*2):2]
        pafs = x[:, (self.nJoints*2):]

        heatmaps = self.sigmoid(heatmaps)

        return heatmaps, pafs

class OpenPoseEstimation(nn.Module):

    '''
    对2D骨架图像进行预测，输出为一个形状为(batch_size, num_frames, num_joints, 3)的张量，其中num_frames是输入视频的帧数，num_joints是25个关节的数量，3是x、y、z三个维度的坐标。
    '''

    def __init__(self, num_joints=25, num_frames=36):  # RuntimeError: shape '[-1, 32, 25, 1]' is invalid for input of size 900
        super(OpenPoseEstimation, self).__init__()
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        # self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(1024, 512)
        self.fc1 = nn.Linear(256, 128)
        # self.fc2 = nn.Linear(512, num_joints * 3)
        self.fc2 = nn.Linear(128, num_joints * 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))
        x = self.pool(x)
        print('pool: ', x.size())
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        print('fc2: ', x.size())
        x = x.view(-1, self.num_frames, self.num_joints, 1)
        return x

class DeepPose(nn.Module):
	"""docstring for DeepPose，nJoints是一个整数，表示人体骨架中关节点的数量。"""
	def __init__(self, nJoints, modelName='resnet50'):
		super(DeepPose, self).__init__()
		self.nJoints = nJoints
		self.block = 'BottleNeck' if (int(modelName[6:]) > 34) else 'BasicBlock'
		self.resnet = getattr(torchvision.models, modelName)(pretrained=True)
		self.resnet.fc = nn.Linear(512 * (4 if self.block == 'BottleNeck' else 1), self.nJoints * 2)
	def forward(self, x):
		return self.resnet(x)



class OpenPoseEstimation(nn.Module):
    def __init__(self, num_classes=2):
        super(OpenPoseEstimation, self).__init__()
        self.num_classes = num_classes
        self.backbone = r3d_18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.num_classes)
        self.conv1 = nn.Conv3d(12, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1)) #96.64
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        print('relu: ', x.size()) # torch.Size([3, 3, 50, 25])
        x = x.unsqueeze(3)
        x = self.backbone(x)
        print('backbone: ', x.size())
        x = self.avgpool(x)
        print('avgpool: ', x.size())
        x = x.view(x.size(0), -1)
        return x

class OpenPoseEstimationMSG3D2(nn.Module):
    def __init__(self, num_classes=2):
        super(OpenPoseEstimationMSG3D2, self).__init__()
        self.num_classes = num_classes
        self.embed_channels_out = 12
        self.openpose = OpenPoseEstimation(self.embed_channels_out)
        self.conv1 = nn.Conv3d(self.embed_channels_out, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.openpose(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x




class PoseEstimation1(nn.Module):
    def __init__(self, num_joints):
        '''其中，num_joints是输出中关键点的数量。'''
        super(PoseEstimation1, self).__init__()

        # 加载预训练的ResNet模型
        self.resnet = models.resnet50(pretrained=True)

        # 修改ResNet的最后一层，使其输出num_joints * 2个通道
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_joints * 2)

        # 定义输出矩阵的形状
        self.num_joints = num_joints
        self.output_shape = (-1, self.num_joints,12, 2)

    def forward(self, x):
        x = self.resnet(x)
        x = x.reshape(self.output_shape)
        return x


import torch
import torch.nn as nn

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
    def __init__(self, in_channels=384, num_joints=25, num_frames=32):
        super(DensePoseEstimation, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 2*num_joints)
        self.num_frames = num_frames

    def forward(self, x):
        N, T, V, M = x.size()
        x = x.view(N, T, V, M)
        x = self.backbone(x)
        x = x.view(N, T, -1)
        x = torch.mean(x, dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(N, self.num_frames, 2, -1)
        return x
