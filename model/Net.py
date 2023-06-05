import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


class LightNet(nn.Module):
    """Backbone for network"""
    def __init__(self, pretrained):
        super(LightNet, self).__init__()

        self.encoder = Encoder()
        # 加载预训练模型并将用到的层提取出来并改名
        if pretrained:
            pretrained_dict = torch.load(pretrained)
            my_state_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                if k.find("layer") != -1:
                    name = k.replace("module.context_path.layer", "layer")
                    my_state_dict[name] = v

            self.encoder.load_state_dict(my_state_dict)

        self.decoder = Decoder()
        self.pyramid_pooling = PyramidPooling(128, 128)
        self.refiner = Refiner()

    def forward(self, x):
        ct_stage1, ct_stage2, ct_stage3, ct_stage4, ct_stage5 = self.encoder(x)
        # (16, 1/2) (32, 1/4) (64, 1/8)  (96, 1/16) (128, 1/32)
        ct_stage6 = self.pyramid_pooling(ct_stage5)  # (128, 1/32)
        y1_out, y2_out, y3_out, y4_out = self.decoder(ct_stage1, ct_stage2, ct_stage3, ct_stage4, ct_stage6)
        final = self.refiner(y1_out)
        return final, y2_out, y3_out, y4_out


class Encoder(nn.Module):
    """Encoder for network"""
    def __init__(self):
        super(Encoder, self).__init__()

        self.layer1 = nn.Sequential(
            ConvBnRelu(3, 16, k=3, s=2, p=1),
            HDRM(16, dilation_level=[1, 2, 3])
        )
        self.layer2 = nn.Sequential(
            EncodeBasicLayer(16, 32, stride=2),
            HDRM(32, dilation_level=[1, 2, 3])
        )
        self.layer3 = nn.Sequential(
            EncodeBasicLayer(32, 64, stride=2),
            HDRM(64, dilation_level=[1, 2, 3]),
            HDRM(64, dilation_level=[1, 2, 3]),
            HDRM(64, dilation_level=[1, 2, 3])
        )
        self.layer4 = nn.Sequential(
            EncodeBasicLayer(64, 96, stride=2),
            HDRM(96, dilation_level=[1, 2, 3]),
            HDRM(96, dilation_level=[1, 2, 3]),
            HDRM(96, dilation_level=[1, 2, 3]),
            HDRM(96, dilation_level=[1, 2, 3]),
            HDRM(96, dilation_level=[1, 2, 3]),
            HDRM(96, dilation_level=[1, 2, 3])
        )
        self.layer5 = nn.Sequential(
            EncodeBasicLayer(96, 128, stride=2),
            HDRM(128, dilation_level=[1, 2]),
            HDRM(128, dilation_level=[1, 2]),
            HDRM(128, dilation_level=[1, 2])
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out1, out2, out3, out4, out5


class Decoder(nn.Module):
    """Decoder for network"""
    def __init__(self):
        super(Decoder, self).__init__()
        self.upsample2time = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.sigmoid = nn.Sigmoid()
        self.step1 = nn.Sequential(
            ConvBnRelu(128, 96, k=3, s=1, p=1),
        )
        self.step2 = nn.Sequential(
            ConvBnRelu(96, 64, k=3, s=1, p=1),
        )
        self.step3 = nn.Sequential(
            ConvBnRelu(64, 32, k=3, s=1, p=1),
        )
        self.step4 = nn.Sequential(
            ConvBnRelu(32, 16, k=3, s=1, p=1),
        )
        self.step5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

        self.y2_out = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.y3_out = nn.Sequential(
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.y4_out = nn.Sequential(
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False),
            nn.Conv2d(96, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # handle y4
        self.x1_to_y4 = SideOutTransfer(in_channel=16, out_channel=96, times=16)  # 112*112 to 7*7
        self.x2_to_y4 = SideOutTransfer(in_channel=32, out_channel=96, times=8)  # 56*56 to 7*7
        self.x3_to_y4 = SideOutTransfer(in_channel=64, out_channel=96, times=4)  # 28*28 to 7*7
        self.x4_to_y4 = SideOutTransfer(in_channel=96, out_channel=96, times=2)  # 14*14 to 7*7
        self.fuse_y4 = DecodeFuse(in_channel=96, out_channel=96)

        # handle y3
        self.x1_to_y3 = SideOutTransfer(in_channel=16, out_channel=64, times=8)  # 112*112 to 14*14
        self.x2_to_y3 = SideOutTransfer(in_channel=32, out_channel=64, times=4)  # 56*56 to 14*14
        self.x3_to_y3 = SideOutTransfer(in_channel=64, out_channel=64, times=2)  # 28*28 to 14*14
        self.x5_to_y3 = SideOutTransfer(in_channel=128, out_channel=64, times=2, magnify=True)  # 7*7 to 14*14
        self.fuse_y3 = DecodeFuse(in_channel=64, out_channel=64)

        # handle y2
        self.x1_to_y2 = SideOutTransfer(in_channel=16, out_channel=32, times=4)  # 112*112 to 28*28
        self.x2_to_y2 = SideOutTransfer(in_channel=32, out_channel=32, times=2)  # 56*56 to 28*28
        self.y4_to_y2 = SideOutTransfer(in_channel=96, out_channel=32, times=2, magnify=True)  # 14*14 to 28*28
        self.x5_to_y2 = SideOutTransfer(in_channel=128, out_channel=32, times=4, magnify=True)  # 7*7 to 28*28
        self.fuse_y2 = DecodeFuse(in_channel=32, out_channel=32)

        # handle y1
        self.x1_to_y1 = SideOutTransfer(in_channel=16, out_channel=16, times=2)  # 112*112 to 56*56
        self.y3_to_y1 = SideOutTransfer(in_channel=64, out_channel=16, times=2, magnify=True)  # 28*28 to 56*56
        self.y4_to_y1 = SideOutTransfer(in_channel=96, out_channel=16, times=4, magnify=True)  # 14*14 to 56*56
        self.x5_to_y1 = SideOutTransfer(in_channel=128, out_channel=16, times=8, magnify=True)  # 7*7 to 56*56
        self.fuse_y1 = DecodeFuse(in_channel=16, out_channel=16)

    def forward(self, x1, x2, x3, x4, x5):
        step1 = self.step1(x5)
        x1_to_y4 = self.x1_to_y4(x1)
        x2_to_y4 = self.x2_to_y4(x2)
        x3_to_y4 = self.x3_to_y4(x3)
        x4_to_y4 = self.x4_to_y4(x4)
        y4 = self.fuse_y4(step1, x1_to_y4, x2_to_y4, x3_to_y4, x4_to_y4)
        y4 = self.upsample2time(y4)

        step2 = self.step2(y4)
        x1_to_y3 = self.x1_to_y3(x1)
        x2_to_y3 = self.x2_to_y3(x2)
        x3_to_y3 = self.x3_to_y3(x3)
        x5_to_y3 = self.x5_to_y3(x5)
        y3 = self.fuse_y3(step2, x1_to_y3, x2_to_y3, x3_to_y3, x5_to_y3)
        y3 = self.upsample2time(y3)

        step3 = self.step3(y3)
        x1_to_y2 = self.x1_to_y2(x1)
        x2_to_y2 = self.x2_to_y2(x2)
        y4_to_y2 = self.y4_to_y2(y4)
        x5_to_y2 = self.x5_to_y2(x5)
        y2 = self.fuse_y2(step3, x1_to_y2, x2_to_y2, y4_to_y2, x5_to_y2)
        y2 = self.upsample2time(y2)

        step4 = self.step4(y2)
        x1_to_y1 = self.x1_to_y1(x1)
        y3_to_y1 = self.y3_to_y1(y3)
        y4_to_y1 = self.y4_to_y1(y4)
        x5_to_y1 = self.x5_to_y1(x5)
        y1 = self.fuse_y1(step4, x1_to_y1, y3_to_y1, y4_to_y1, x5_to_y1)
        y1 = self.upsample2time(y1)

        y1_out = self.step5(y1)
        y2_out = self.y2_out(y2)
        y3_out = self.y3_out(y3)
        y4_out = self.y4_out(y4)

        return y1_out, y2_out, y3_out, y4_out


class Refiner(nn.Module):
    """Refiner for network"""
    def __init__(self):
        super(Refiner, self).__init__()
        self.refine = nn.Sequential(
            ConvBnRelu(in_channel=1, out_channel=4, s=2),  # 112*112
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        refine = self.refine(x)
        out = self.sigmoid(x+refine)
        return out


class HDRM(nn.Module):
    """Hybrid Dilated Residual Module for each encode layer"""
    def __init__(self, channel, dilation_level=[1,2,4,8], reduce_factor=4):
        super(HDRM, self).__init__()
        self.planes = channel
        self.dilation_level = dilation_level
        self.conv = EncodeBasicLayer(channel, channel, stride=1)
        self.branches = nn.ModuleList([
            EncodeBasicLayer(channel, channel, stride=1, dilation=d) for d in dilation_level
        ])

        ### ChannelGate
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = ConvBnRelu(channel, channel, 1, 1, 0, bn=True, relu=True)
        self.fc2 = nn.Conv2d(channel, (len(self.dilation_level) + 1) * channel, 1, 1, 0, bias=False)
        self.fuse = ConvBnRelu(channel, channel, k=1, s=1, p=0, relu=False)
        ### SpatialGate
        self.convs = nn.Sequential(
            ConvBnRelu(channel, channel // 4, 1, 1, 0, bn=True, relu=True),
            EncodeBasicLayer(channel // 4, channel // 4, stride=1, dilation=2),
            EncodeBasicLayer(channel // 4, channel // 4, stride=1, dilation=4),
            nn.Conv2d(channel // 4, 1, 1, 1, 0, bias=False)
        )

    def forward(self, x):
        conv = self.conv(x)
        brs = [branch(conv) for branch in self.branches]
        brs.append(conv)
        gather = sum(brs)

        ### ChannelGate
        d = self.gap(gather)
        d = self.fc2(self.fc1(d))
        d = torch.unsqueeze(d, dim=1).view(-1, len(self.dilation_level) + 1, self.planes, 1, 1)

        ### SpatialGate
        s = self.convs(gather).unsqueeze(1)

        ### Fuse two gates
        f = d * s
        f = F.softmax(f, dim=1)

        return self.fuse(sum([brs[i] * f[:, i, ...] for i in range(len(self.dilation_level) + 1)])) + x


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


class EncodeBasicLayer(nn.Module):
    """Encode Basic layer"""
    def __init__(self, in_channel, out_channel, stride=1, dilation=1):
        super(EncodeBasicLayer, self).__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            ConvBnRelu(in_channel, out_channel, k=1, s=1, p=0)
        )

    def forward(self, x):
        return self.conv(x)


class SideOutTransfer(nn.Module):
    """
    Side Out Transfer for decoder:
    resize the input tensor
    """
    def __init__(self, in_channel, out_channel, times=1, magnify=False):
        super(SideOutTransfer, self).__init__()

        self.transfer = nn.Sequential(
            nn.Upsample(scale_factor=times, mode="bilinear", align_corners=False) if magnify else nn.MaxPool2d(times, times, ceil_mode=True),
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.transfer(x)


class DecodeFuse(nn.Module):
    """Decode Fuse module"""
    def __init__(self, in_channel, out_channel):
        super(DecodeFuse, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel//4),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel//4),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel//4),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel//4),
            nn.ReLU(inplace=True),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, last, input1, input2, input3, input4):
        input1 = self.conv1(input1)
        input2 = self.conv2(input2)
        input3 = self.conv3(input3)
        input4 = self.conv4(input4)
        merge = self.merge(torch.cat((input1, input2, input3, input4), dim=1))
        return last+merge


class PyramidPooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PyramidPooling, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = ConvBnRelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = ConvBnRelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = ConvBnRelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = ConvBnRelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = ConvBnRelu(in_channel*2, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
        feat2 = interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
        feat3 = interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
        feat4 = interpolate(self.conv4(F.adaptive_avg_pool2d(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)

        return x


interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)
