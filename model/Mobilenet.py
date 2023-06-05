from torchvision import models
from torch import nn
import torch
from torchsummary import summary

'''
mobilenetv2 修改最后的全连接层为反卷积
用作x光图对比试验
'''

class MobileNet(nn.Module):

    def __init__(self, pretrained=False):
        super(MobileNet, self).__init__()
        mobilenet = models.mobilenet_v2()
        # print(mobilenet)
        # exit()
        net = mobilenet.features

        # 修改最后的全连接层
        net[18] = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 14*14
            ConvBnRelu(320, 160, k=3, s=1, p=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 28*28
            ConvBnRelu(160, 80, k=3, s=1, p=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 56*56
            ConvBnRelu(80, 40, k=3, s=1, p=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 112*112
            ConvBnRelu(40, 20, k=3, s=1, p=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 224*224
            ConvBnRelu(20, 1, k=3, s=1, p=1),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
        )

        self.sigmoid = nn.Sigmoid()
        self.net = net

    def forward(self, x):
        x = self.net(x)
        x = self.sigmoid(x)
        return x


class MobileNet3(nn.Module):

    def __init__(self, pretrained=False):
        super(MobileNet3, self).__init__()
        mobilenet = models.mobilenet_v3_small()
        net = mobilenet.features

        net.add_module('out', nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 14*14
            ConvBnRelu(576, 288, k=3, s=1, p=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 28*28
            ConvBnRelu(288, 144, k=3, s=1, p=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 56*56
            ConvBnRelu(144, 72, k=3, s=1, p=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 112*112
            ConvBnRelu(72, 36, k=3, s=1, p=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 224*224
            ConvBnRelu(36, 1, k=3, s=1, p=1),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
        ))

        self.sigmoid = nn.Sigmoid()
        self.net = net

    def forward(self, x):
        x = self.net(x)
        x = self.sigmoid(x)
        return x


class ConvBnRelu(nn.Module):
    """Convolution + BatchNormalization + Relu"""
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(ConvBnRelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class TNet(nn.Module):

    def __init__(self, pretrained=False):
        super(TNet, self).__init__()

        self.sigmoid = nn.Sigmoid()

        self.net = nn.Sequential(
            ConvBnRelu(in_channel=3, out_channel=4, s=2),  # 112*112
            ConvBnRelu(in_channel=4, out_channel=8, s=2),  # 56*56
            ConvBnRelu(in_channel=8, out_channel=16, s=2),  # 28*28
            ConvBnRelu(in_channel=16, out_channel=24, s=2),  # 14*14
            ConvBnRelu(in_channel=24, out_channel=36, s=2),  # 7*7
            ConvBnRelu(in_channel=36, out_channel=36, s=1),  # 7*7
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 14*14
            ConvBnRelu(36, 24, k=3, s=1, p=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 28*28
            ConvBnRelu(24, 16, k=3, s=1, p=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 56*56
            ConvBnRelu(16, 8, k=3, s=1, p=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 112*112
            ConvBnRelu(8, 4, k=3, s=1, p=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 224*224
            ConvBnRelu(4, 1, k=3, s=1, p=1),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.sigmoid(x)
        return x