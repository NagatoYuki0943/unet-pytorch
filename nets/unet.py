"""
加强特征提取网络
将低层数据上采样和上层数据cat到一起,再进行两次卷积即可
"""

import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16


class unetUp(nn.Module):
    """
    上采样
    将低层数据上采样和上层数据cat到一起,再进行两次卷积即可
    """
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()

        # 特征融合
        # in_size = 下层通道+上层通道
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        # 宽高变为2倍
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        """
        inputs1: 上层
        inputs2: 下层,要上采样
        """
        # 上采样
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        # 卷积
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        # in_filters = 下层通道+上层通道
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        # 上采样并拼接后最终调整为这些维度
        out_filters = [64, 128, 256, 512]

        # vgg upsampling 从下到上
        # [32, 32, 512] up+cat [64, 64, 512] => [64, 64, 512]
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # [64, 64, 512] up+cat [128,128,256] => [128,128,256]
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # [128,128,256] up+cat [256,256,128] => [256,256,128]
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # [256,256,128] up+cat [512,512, 64] => [512,512, 64]
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        # resnet50还要在进行一次上采样,因为resnet的feat1大小经过了压缩一次 [256,256,64] => [512,512,64]
        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        # 最终通道变为分类数
        # [512,512,64] => [512,512,num_classes]
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            # 宽高 512 256 128 64 32
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            # 宽高 256 128 64 32 16
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        #                     上层   下层
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)   # out: [512,512,64]

        # resnet才使用
        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
