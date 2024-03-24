

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

from network.SemiModel import GCM,aggregation_final, aggregation_init,Refine,ChannelAttention,SpatialAttention
from network.SemiModel import BasicConv2d



# 消融实验：代替GCM模块，保证拿掉GCM后通道数一致
class replace_GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(replace_GCM, self).__init__()
        self.relu = nn.ReLU(True)
        # self.branch0 = nn.Sequential(
        #     BasicConv2d(in_channel, out_channel, 1),
        # )
        # self.branch1 = nn.Sequential(
        #     BasicConv2d(in_channel, out_channel, 1),
        #     BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
        #     BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
        #     BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        # )
        # self.branch2 = nn.Sequential(
        #     BasicConv2d(in_channel, out_channel, 1),
        #     BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
        #     BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
        #     BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        # )
        # self.branch3 = nn.Sequential(
        #     BasicConv2d(in_channel, out_channel, 1),
        #     BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
        #     BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
        #     BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        # )
        self.conv_cat = BasicConv2d(in_channel, out_channel, 3, padding=1)
        # self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x = self.conv_cat(x)
        x = self.relu(x)
        # x0 = self.branch0(x)
        # x1 = self.branch1(x)
        # x2 = self.branch2(x)
        # x3 = self.branch3(x)
        # x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        # x = self.relu(x_cat + self.conv_res(x))
        return x

class SemiModel_without_GCM_3_4_5(nn.Module):
    def __init__(self, channel=32):
        super(SemiModel_without_GCM_3_4_5, self).__init__()

        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512

        self.rfb3 = replace_GCM(256, channel)
        self.rfb4 = replace_GCM(512, channel)
        self.rfb5 = replace_GCM(512, channel)
        # self.rfb3 = GCM(256, channel)
        # self.rfb4 = GCM(512, channel)
        # self.rfb5 = GCM(512, channel)
        self.agg1 = aggregation_init(channel)

        self.rfb1 = GCM(64, channel)
        self.rfb2 = GCM(128, channel)
        self.rfb3_2 = GCM(channel, channel)
        self.agg2 = aggregation_final(channel)

        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.HA = Refine()

        self.atten_A_channel_1 =ChannelAttention(64)
        self.atten_A_channel_2 =ChannelAttention(128)
        self.atten_A_channel_3 =ChannelAttention(256)
        self.atten_A_channel_4 =ChannelAttention(512)
        self.atten_A_channel_5 =ChannelAttention(512)

        self.atten_A_spatial_1 =SpatialAttention()
        self.atten_A_spatial_2 =SpatialAttention()
        self.atten_A_spatial_3 =SpatialAttention()
        self.atten_A_spatial_4 =SpatialAttention()
        self.atten_A_spatial_5 =SpatialAttention()

        self.atten_B_channel_1 =ChannelAttention(64)
        self.atten_B_channel_2 =ChannelAttention(128)
        self.atten_B_channel_3 =ChannelAttention(256)
        self.atten_B_channel_4 =ChannelAttention(512)
        self.atten_B_channel_5 =ChannelAttention(512)

        self.atten_B_spatial_1 =SpatialAttention()
        self.atten_B_spatial_2 =SpatialAttention()
        self.atten_B_spatial_3 =SpatialAttention()
        self.atten_B_spatial_4 =SpatialAttention()
        self.atten_B_spatial_5 =SpatialAttention()

        self.agant1 = self._make_agant_layer(32 *3, 32 *2)
        self.agant2 = self._make_agant_layer(32 *2, 32)
        self.out_conv = nn.Conv2d(32 *1, 1, kernel_size=1, stride=1, bias=True)

    def forward(self, A, B):
        layer1_A = self.inc(A)
        layer2_A = self.down1(layer1_A)
        layer3_A = self.down2(layer2_A)
        layer4_A = self.down3(layer3_A)
        layer5_A = self.down4(layer4_A)

        layer1_B = self.inc(B)
        layer2_B = self.down1(layer1_B)
        layer3_B = self.down2(layer2_B)
        layer4_B = self.down3(layer3_B)
        layer5_B = self.down4(layer4_B)

        layer1_A = layer1_A.mul(self.atten_A_channel_1(layer1_A))
        layer1_A = layer1_A.mul(self.atten_A_spatial_1(layer1_A))

        layer1_B = layer1_B.mul(self.atten_B_channel_1(layer1_B))
        layer1_B = layer1_B.mul(self.atten_B_spatial_1(layer1_B))
        layer1 = layer1_A + layer1_B

        layer2_A = layer2_A.mul(self.atten_A_channel_2(layer2_A))
        layer2_A = layer2_A.mul(self.atten_A_spatial_2(layer2_A))

        layer2_B = layer2_B.mul(self.atten_B_channel_2(layer2_B))
        layer2_B = layer2_B.mul(self.atten_B_spatial_2(layer2_B))
        layer2 = layer2_A + layer2_B

        layer3_A = layer3_A.mul(self.atten_A_channel_3(layer3_A))
        layer3_A = layer3_A.mul(self.atten_A_spatial_3(layer3_A))

        layer3_B = layer3_B.mul(self.atten_B_channel_3(layer3_B))
        layer3_B = layer3_B.mul(self.atten_B_spatial_3(layer3_B))
        layer3 = layer3_A + layer3_B

        layer4_A = layer4_A.mul(self.atten_A_channel_4(layer4_A))
        layer4_A = layer4_A.mul(self.atten_A_spatial_4(layer4_A))

        layer4_B = layer4_B.mul(self.atten_B_channel_4(layer4_B))
        layer4_B = layer4_B.mul(self.atten_B_spatial_4(layer4_B))
        layer4 = layer4_A + layer4_B

        layer5_A = layer5_A.mul(self.atten_A_channel_5(layer5_A))
        layer5_A = layer5_A.mul(self.atten_A_spatial_5(layer5_A))

        layer5_B = layer5_B.mul(self.atten_B_channel_5(layer5_B))
        layer5_B = layer5_B.mul(self.atten_B_spatial_5(layer5_B))
        layer5 = layer5_A + layer5_B

        layer3 = self.rfb3(layer3)
        layer4 = self.rfb4(layer4)
        layer5 = self.rfb5(layer5)
        attention_map = self.agg1(layer5 ,layer4 ,layer3)

        layer1 ,layer2 ,layer3 = self.HA(attention_map.sigmoid(), layer1 ,layer2 ,layer3)

        layer1 = self.rfb1(layer1)
        layer2 = self.rfb2(layer2)
        layer3 = self.rfb3_2(layer3)

        y = self.agg2(layer3, layer2, layer1)  # *4

        y =self.agant1(y)
        y = self.agant2(y)
        y = self.out_conv(y)

        return F.interpolate(attention_map, size=A.size()[2:], mode='bilinear', align_corners=True), F.interpolate(y,
                                                                                                                   size=A.size()[
                                                                                                                        2:],
                                                                                                                   mode='bilinear',
                                                                                                                   align_corners=True)

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

class SemiModel_without_GCM_1_2_3(nn.Module):
    def __init__(self, channel=32):
        super(SemiModel_without_GCM_1_2_3, self).__init__()

        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512


        self.rfb3 = GCM(256, channel)
        self.rfb4 = GCM(512, channel)
        self.rfb5 = GCM(512, channel)
        self.agg1 = aggregation_init(channel)

        self.rfb1 = replace_GCM(64, channel)
        self.rfb2 = replace_GCM(128, channel)
        self.rfb3_2 = replace_GCM(channel, channel)
        # self.rfb1 = GCM(64, channel)
        # self.rfb2 = GCM(128, channel)
        # self.rfb3_2 = GCM(channel, channel)
        self.agg2 = aggregation_final(channel)

        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.HA = Refine()

        self.atten_A_channel_1 =ChannelAttention(64)
        self.atten_A_channel_2 =ChannelAttention(128)
        self.atten_A_channel_3 =ChannelAttention(256)
        self.atten_A_channel_4 =ChannelAttention(512)
        self.atten_A_channel_5 =ChannelAttention(512)

        self.atten_A_spatial_1 =SpatialAttention()
        self.atten_A_spatial_2 =SpatialAttention()
        self.atten_A_spatial_3 =SpatialAttention()
        self.atten_A_spatial_4 =SpatialAttention()
        self.atten_A_spatial_5 =SpatialAttention()

        self.atten_B_channel_1 =ChannelAttention(64)
        self.atten_B_channel_2 =ChannelAttention(128)
        self.atten_B_channel_3 =ChannelAttention(256)
        self.atten_B_channel_4 =ChannelAttention(512)
        self.atten_B_channel_5 =ChannelAttention(512)

        self.atten_B_spatial_1 =SpatialAttention()
        self.atten_B_spatial_2 =SpatialAttention()
        self.atten_B_spatial_3 =SpatialAttention()
        self.atten_B_spatial_4 =SpatialAttention()
        self.atten_B_spatial_5 =SpatialAttention()

        self.agant1 = self._make_agant_layer(32 *3, 32 *2)
        self.agant2 = self._make_agant_layer(32 *2, 32)
        self.out_conv = nn.Conv2d(32 *1, 1, kernel_size=1, stride=1, bias=True)

    def forward(self, A, B):
        layer1_A = self.inc(A)
        layer2_A = self.down1(layer1_A)
        layer3_A = self.down2(layer2_A)
        layer4_A = self.down3(layer3_A)
        layer5_A = self.down4(layer4_A)

        layer1_B = self.inc(B)
        layer2_B = self.down1(layer1_B)
        layer3_B = self.down2(layer2_B)
        layer4_B = self.down3(layer3_B)
        layer5_B = self.down4(layer4_B)

        layer1_A = layer1_A.mul(self.atten_A_channel_1(layer1_A))
        layer1_A = layer1_A.mul(self.atten_A_spatial_1(layer1_A))

        layer1_B = layer1_B.mul(self.atten_B_channel_1(layer1_B))
        layer1_B = layer1_B.mul(self.atten_B_spatial_1(layer1_B))
        layer1 = layer1_A + layer1_B

        layer2_A = layer2_A.mul(self.atten_A_channel_2(layer2_A))
        layer2_A = layer2_A.mul(self.atten_A_spatial_2(layer2_A))

        layer2_B = layer2_B.mul(self.atten_B_channel_2(layer2_B))
        layer2_B = layer2_B.mul(self.atten_B_spatial_2(layer2_B))
        layer2 = layer2_A + layer2_B

        layer3_A = layer3_A.mul(self.atten_A_channel_3(layer3_A))
        layer3_A = layer3_A.mul(self.atten_A_spatial_3(layer3_A))

        layer3_B = layer3_B.mul(self.atten_B_channel_3(layer3_B))
        layer3_B = layer3_B.mul(self.atten_B_spatial_3(layer3_B))
        layer3 = layer3_A + layer3_B

        layer4_A = layer4_A.mul(self.atten_A_channel_4(layer4_A))
        layer4_A = layer4_A.mul(self.atten_A_spatial_4(layer4_A))

        layer4_B = layer4_B.mul(self.atten_B_channel_4(layer4_B))
        layer4_B = layer4_B.mul(self.atten_B_spatial_4(layer4_B))
        layer4 = layer4_A + layer4_B

        layer5_A = layer5_A.mul(self.atten_A_channel_5(layer5_A))
        layer5_A = layer5_A.mul(self.atten_A_spatial_5(layer5_A))

        layer5_B = layer5_B.mul(self.atten_B_channel_5(layer5_B))
        layer5_B = layer5_B.mul(self.atten_B_spatial_5(layer5_B))
        layer5 = layer5_A + layer5_B

        layer3 = self.rfb3(layer3)
        layer4 = self.rfb4(layer4)
        layer5 = self.rfb5(layer5)
        attention_map = self.agg1(layer5 ,layer4 ,layer3)

        layer1 ,layer2 ,layer3 = self.HA(attention_map.sigmoid(), layer1 ,layer2 ,layer3)

        layer1 = self.rfb1(layer1)
        layer2 = self.rfb2(layer2)
        layer3 = self.rfb3_2(layer3)

        y = self.agg2(layer3, layer2, layer1)  # *4

        y =self.agant1(y)
        y = self.agant2(y)
        y = self.out_conv(y)

        return F.interpolate(attention_map, size=A.size()[2:], mode='bilinear', align_corners=True), F.interpolate(y,
                                                                                                                   size=A.size()[
                                                                                                                        2:],
                                                                                                                   mode='bilinear',
                                                                                                                   align_corners=True)

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

class SemiModel_without_all_GCM(nn.Module):
    def __init__(self, channel=32):
        super(SemiModel_without_all_GCM, self).__init__()

        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512

        self.rfb3 = replace_GCM(256, channel)
        self.rfb4 = replace_GCM(512, channel)
        self.rfb5 = replace_GCM(512, channel)
        # self.rfb3 = GCM(256, channel)
        # self.rfb4 = GCM(512, channel)
        # self.rfb5 = GCM(512, channel)
        self.agg1 = aggregation_init(channel)

        self.rfb1 = replace_GCM(64, channel)
        self.rfb2 = replace_GCM(128, channel)
        self.rfb3_2 = replace_GCM(channel, channel)
        # self.rfb1 = GCM(64, channel)
        # self.rfb2 = GCM(128, channel)
        # self.rfb3_2 = GCM(channel, channel)
        self.agg2 = aggregation_final(channel)

        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.HA = Refine()

        self.atten_A_channel_1 =ChannelAttention(64)
        self.atten_A_channel_2 =ChannelAttention(128)
        self.atten_A_channel_3 =ChannelAttention(256)
        self.atten_A_channel_4 =ChannelAttention(512)
        self.atten_A_channel_5 =ChannelAttention(512)

        self.atten_A_spatial_1 =SpatialAttention()
        self.atten_A_spatial_2 =SpatialAttention()
        self.atten_A_spatial_3 =SpatialAttention()
        self.atten_A_spatial_4 =SpatialAttention()
        self.atten_A_spatial_5 =SpatialAttention()

        self.atten_B_channel_1 =ChannelAttention(64)
        self.atten_B_channel_2 =ChannelAttention(128)
        self.atten_B_channel_3 =ChannelAttention(256)
        self.atten_B_channel_4 =ChannelAttention(512)
        self.atten_B_channel_5 =ChannelAttention(512)

        self.atten_B_spatial_1 =SpatialAttention()
        self.atten_B_spatial_2 =SpatialAttention()
        self.atten_B_spatial_3 =SpatialAttention()
        self.atten_B_spatial_4 =SpatialAttention()
        self.atten_B_spatial_5 =SpatialAttention()

        self.agant1 = self._make_agant_layer(32 *3, 32 *2)
        self.agant2 = self._make_agant_layer(32 *2, 32)
        self.out_conv = nn.Conv2d(32 *1, 1, kernel_size=1, stride=1, bias=True)

    def forward(self, A, B):
        layer1_A = self.inc(A)
        layer2_A = self.down1(layer1_A)
        layer3_A = self.down2(layer2_A)
        layer4_A = self.down3(layer3_A)
        layer5_A = self.down4(layer4_A)

        layer1_B = self.inc(B)
        layer2_B = self.down1(layer1_B)
        layer3_B = self.down2(layer2_B)
        layer4_B = self.down3(layer3_B)
        layer5_B = self.down4(layer4_B)

        layer1_A = layer1_A.mul(self.atten_A_channel_1(layer1_A))
        layer1_A = layer1_A.mul(self.atten_A_spatial_1(layer1_A))

        layer1_B = layer1_B.mul(self.atten_B_channel_1(layer1_B))
        layer1_B = layer1_B.mul(self.atten_B_spatial_1(layer1_B))
        layer1 = layer1_A + layer1_B

        layer2_A = layer2_A.mul(self.atten_A_channel_2(layer2_A))
        layer2_A = layer2_A.mul(self.atten_A_spatial_2(layer2_A))

        layer2_B = layer2_B.mul(self.atten_B_channel_2(layer2_B))
        layer2_B = layer2_B.mul(self.atten_B_spatial_2(layer2_B))
        layer2 = layer2_A + layer2_B

        layer3_A = layer3_A.mul(self.atten_A_channel_3(layer3_A))
        layer3_A = layer3_A.mul(self.atten_A_spatial_3(layer3_A))

        layer3_B = layer3_B.mul(self.atten_B_channel_3(layer3_B))
        layer3_B = layer3_B.mul(self.atten_B_spatial_3(layer3_B))
        layer3 = layer3_A + layer3_B

        layer4_A = layer4_A.mul(self.atten_A_channel_4(layer4_A))
        layer4_A = layer4_A.mul(self.atten_A_spatial_4(layer4_A))

        layer4_B = layer4_B.mul(self.atten_B_channel_4(layer4_B))
        layer4_B = layer4_B.mul(self.atten_B_spatial_4(layer4_B))
        layer4 = layer4_A + layer4_B

        layer5_A = layer5_A.mul(self.atten_A_channel_5(layer5_A))
        layer5_A = layer5_A.mul(self.atten_A_spatial_5(layer5_A))

        layer5_B = layer5_B.mul(self.atten_B_channel_5(layer5_B))
        layer5_B = layer5_B.mul(self.atten_B_spatial_5(layer5_B))
        layer5 = layer5_A + layer5_B

        layer3 = self.rfb3(layer3)
        layer4 = self.rfb4(layer4)
        layer5 = self.rfb5(layer5)
        attention_map = self.agg1(layer5 ,layer4 ,layer3)

        layer1 ,layer2 ,layer3 = self.HA(attention_map.sigmoid(), layer1 ,layer2 ,layer3)

        layer1 = self.rfb1(layer1)
        layer2 = self.rfb2(layer2)
        layer3 = self.rfb3_2(layer3)

        y = self.agg2(layer3, layer2, layer1)  # *4

        y =self.agant1(y)
        y = self.agant2(y)
        y = self.out_conv(y)

        return F.interpolate(attention_map, size=A.size()[2:], mode='bilinear', align_corners=True), F.interpolate(y,
                                                                                                                   size=A.size()[
                                                                                                                        2:],
                                                                                                                   mode='bilinear',
                                                                                                                   align_corners=True)

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

# Refine用attention map进行refine，这里消融实验直接去掉refine
class SemiModel_without_Refine(nn.Module):
    def __init__(self, channel=32):
        super(SemiModel_without_Refine, self).__init__()

        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512

        self.rfb3 = GCM(256, channel)
        self.rfb4 = GCM(512, channel)
        self.rfb5 = GCM(512, channel)
        self.agg1 = aggregation_init(channel)

        self.rfb1 = GCM(64, channel)
        self.rfb2 = GCM(128, channel)
        self.rfb3_2 = GCM(channel, channel)
        self.agg2 = aggregation_final(channel)

        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        # self.HA = Refine()

        self.atten_A_channel_1 = ChannelAttention(64)
        self.atten_A_channel_2 = ChannelAttention(128)
        self.atten_A_channel_3 = ChannelAttention(256)
        self.atten_A_channel_4 = ChannelAttention(512)
        self.atten_A_channel_5 = ChannelAttention(512)

        self.atten_A_spatial_1 = SpatialAttention()
        self.atten_A_spatial_2 = SpatialAttention()
        self.atten_A_spatial_3 = SpatialAttention()
        self.atten_A_spatial_4 = SpatialAttention()
        self.atten_A_spatial_5 = SpatialAttention()

        self.atten_B_channel_1 = ChannelAttention(64)
        self.atten_B_channel_2 = ChannelAttention(128)
        self.atten_B_channel_3 = ChannelAttention(256)
        self.atten_B_channel_4 = ChannelAttention(512)
        self.atten_B_channel_5 = ChannelAttention(512)

        self.atten_B_spatial_1 = SpatialAttention()
        self.atten_B_spatial_2 = SpatialAttention()
        self.atten_B_spatial_3 = SpatialAttention()
        self.atten_B_spatial_4 = SpatialAttention()
        self.atten_B_spatial_5 = SpatialAttention()

        self.agant1 = self._make_agant_layer(32 * 3, 32 * 2)
        self.agant2 = self._make_agant_layer(32 * 2, 32)
        self.out_conv = nn.Conv2d(32 * 1, 1, kernel_size=1, stride=1, bias=True)

    def forward(self, A, B):
        layer1_A = self.inc(A)
        layer2_A = self.down1(layer1_A)
        layer3_A = self.down2(layer2_A)
        layer4_A = self.down3(layer3_A)
        layer5_A = self.down4(layer4_A)

        layer1_B = self.inc(B)
        layer2_B = self.down1(layer1_B)
        layer3_B = self.down2(layer2_B)
        layer4_B = self.down3(layer3_B)
        layer5_B = self.down4(layer4_B)

        layer1_A = layer1_A.mul(self.atten_A_channel_1(layer1_A))
        layer1_A = layer1_A.mul(self.atten_A_spatial_1(layer1_A))

        layer1_B = layer1_B.mul(self.atten_B_channel_1(layer1_B))
        layer1_B = layer1_B.mul(self.atten_B_spatial_1(layer1_B))
        layer1 = layer1_A + layer1_B

        layer2_A = layer2_A.mul(self.atten_A_channel_2(layer2_A))
        layer2_A = layer2_A.mul(self.atten_A_spatial_2(layer2_A))

        layer2_B = layer2_B.mul(self.atten_B_channel_2(layer2_B))
        layer2_B = layer2_B.mul(self.atten_B_spatial_2(layer2_B))
        layer2 = layer2_A + layer2_B

        layer3_A = layer3_A.mul(self.atten_A_channel_3(layer3_A))
        layer3_A = layer3_A.mul(self.atten_A_spatial_3(layer3_A))

        layer3_B = layer3_B.mul(self.atten_B_channel_3(layer3_B))
        layer3_B = layer3_B.mul(self.atten_B_spatial_3(layer3_B))
        layer3 = layer3_A + layer3_B

        layer4_A = layer4_A.mul(self.atten_A_channel_4(layer4_A))
        layer4_A = layer4_A.mul(self.atten_A_spatial_4(layer4_A))

        layer4_B = layer4_B.mul(self.atten_B_channel_4(layer4_B))
        layer4_B = layer4_B.mul(self.atten_B_spatial_4(layer4_B))
        layer4 = layer4_A + layer4_B

        layer5_A = layer5_A.mul(self.atten_A_channel_5(layer5_A))
        layer5_A = layer5_A.mul(self.atten_A_spatial_5(layer5_A))

        layer5_B = layer5_B.mul(self.atten_B_channel_5(layer5_B))
        layer5_B = layer5_B.mul(self.atten_B_spatial_5(layer5_B))
        layer5 = layer5_A + layer5_B

        layer3 = self.rfb3(layer3)
        layer4 = self.rfb4(layer4)
        layer5 = self.rfb5(layer5)
        attention_map = self.agg1(layer5, layer4, layer3)

        # layer1, layer2, layer3 = self.HA(attention_map.sigmoid(), layer1, layer2, layer3)

        layer1 = self.rfb1(layer1)
        layer2 = self.rfb2(layer2)
        layer3 = self.rfb3_2(layer3)

        y = self.agg2(layer3, layer2, layer1)  # *4

        y = self.agant1(y)
        y = self.agant2(y)
        y = self.out_conv(y)

        return F.interpolate(attention_map, size=A.size()[2:], mode='bilinear', align_corners=True), F.interpolate(y,
                                                                                                                   size=A.size()[
                                                                                                                        2:],
                                                                                                                   mode='bilinear',
                                                                                                                   align_corners=True)

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers


class replace_aggregation_init(nn.Module):
    def __init__(self, channel):
        super(replace_aggregation_init, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        # self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        # self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        # self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        # self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        #
        # self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        # self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1 = self.upsample(x1)
        x1 = self.upsample(x1)
        x2 = self.upsample(x2)


        # x1_1 = x1
        # x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        # x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
        #        * self.conv_upsample3(self.upsample(x2)) * x3
        #
        # x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        # x2_2 = self.conv_concat2(x2_2)
        #
        # x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        # x3_2 = self.conv_concat3(x3_2)

        x = torch.cat(( x1, x2, x3), 1)
        x = self.conv4(x)
        x = self.conv5(x)

        return x
class SemiModel_without_agg_init(nn.Module):
    def __init__(self, channel=32):
        super(SemiModel_without_agg_init, self).__init__()

        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512

        self.rfb3 = GCM(256, channel)
        self.rfb4 = GCM(512, channel)
        self.rfb5 = GCM(512, channel)
        # self.agg1 = aggregation_init(channel)
        self.agg1 = replace_aggregation_init(channel)



        self.rfb1 = GCM(64, channel)
        self.rfb2 = GCM(128, channel)
        self.rfb3_2 = GCM(channel, channel)
        self.agg2 = aggregation_final(channel)

        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.HA = Refine()

        self.atten_A_channel_1 = ChannelAttention(64)
        self.atten_A_channel_2 = ChannelAttention(128)
        self.atten_A_channel_3 = ChannelAttention(256)
        self.atten_A_channel_4 = ChannelAttention(512)
        self.atten_A_channel_5 = ChannelAttention(512)

        self.atten_A_spatial_1 = SpatialAttention()
        self.atten_A_spatial_2 = SpatialAttention()
        self.atten_A_spatial_3 = SpatialAttention()
        self.atten_A_spatial_4 = SpatialAttention()
        self.atten_A_spatial_5 = SpatialAttention()

        self.atten_B_channel_1 = ChannelAttention(64)
        self.atten_B_channel_2 = ChannelAttention(128)
        self.atten_B_channel_3 = ChannelAttention(256)
        self.atten_B_channel_4 = ChannelAttention(512)
        self.atten_B_channel_5 = ChannelAttention(512)

        self.atten_B_spatial_1 = SpatialAttention()
        self.atten_B_spatial_2 = SpatialAttention()
        self.atten_B_spatial_3 = SpatialAttention()
        self.atten_B_spatial_4 = SpatialAttention()
        self.atten_B_spatial_5 = SpatialAttention()

        self.agant1 = self._make_agant_layer(32 * 3, 32 * 2)
        self.agant2 = self._make_agant_layer(32 * 2, 32)
        self.out_conv = nn.Conv2d(32 * 1, 1, kernel_size=1, stride=1, bias=True)

    def forward(self, A, B):
        layer1_A = self.inc(A)
        layer2_A = self.down1(layer1_A)
        layer3_A = self.down2(layer2_A)
        layer4_A = self.down3(layer3_A)
        layer5_A = self.down4(layer4_A)

        layer1_B = self.inc(B)
        layer2_B = self.down1(layer1_B)
        layer3_B = self.down2(layer2_B)
        layer4_B = self.down3(layer3_B)
        layer5_B = self.down4(layer4_B)

        layer1_A = layer1_A.mul(self.atten_A_channel_1(layer1_A))
        layer1_A = layer1_A.mul(self.atten_A_spatial_1(layer1_A))

        layer1_B = layer1_B.mul(self.atten_B_channel_1(layer1_B))
        layer1_B = layer1_B.mul(self.atten_B_spatial_1(layer1_B))
        layer1 = layer1_A + layer1_B

        layer2_A = layer2_A.mul(self.atten_A_channel_2(layer2_A))
        layer2_A = layer2_A.mul(self.atten_A_spatial_2(layer2_A))

        layer2_B = layer2_B.mul(self.atten_B_channel_2(layer2_B))
        layer2_B = layer2_B.mul(self.atten_B_spatial_2(layer2_B))
        layer2 = layer2_A + layer2_B

        layer3_A = layer3_A.mul(self.atten_A_channel_3(layer3_A))
        layer3_A = layer3_A.mul(self.atten_A_spatial_3(layer3_A))

        layer3_B = layer3_B.mul(self.atten_B_channel_3(layer3_B))
        layer3_B = layer3_B.mul(self.atten_B_spatial_3(layer3_B))
        layer3 = layer3_A + layer3_B

        layer4_A = layer4_A.mul(self.atten_A_channel_4(layer4_A))
        layer4_A = layer4_A.mul(self.atten_A_spatial_4(layer4_A))

        layer4_B = layer4_B.mul(self.atten_B_channel_4(layer4_B))
        layer4_B = layer4_B.mul(self.atten_B_spatial_4(layer4_B))
        layer4 = layer4_A + layer4_B

        layer5_A = layer5_A.mul(self.atten_A_channel_5(layer5_A))
        layer5_A = layer5_A.mul(self.atten_A_spatial_5(layer5_A))

        layer5_B = layer5_B.mul(self.atten_B_channel_5(layer5_B))
        layer5_B = layer5_B.mul(self.atten_B_spatial_5(layer5_B))
        layer5 = layer5_A + layer5_B

        layer3 = self.rfb3(layer3)
        layer4 = self.rfb4(layer4)
        layer5 = self.rfb5(layer5)
        attention_map = self.agg1(layer5, layer4, layer3)

        layer1, layer2, layer3 = self.HA(attention_map.sigmoid(), layer1, layer2, layer3)

        layer1 = self.rfb1(layer1)
        layer2 = self.rfb2(layer2)
        layer3 = self.rfb3_2(layer3)

        y = self.agg2(layer3, layer2, layer1)  # *4

        y = self.agant1(y)
        y = self.agant2(y)
        y = self.out_conv(y)

        return F.interpolate(attention_map, size=A.size()[2:], mode='bilinear', align_corners=True), F.interpolate(y,
                                                                                                                   size=A.size()[
                                                                                                                        2:],
                                                                                                                   mode='bilinear',
                                                                                                                   align_corners=True)

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers


class replace_aggregation_final(nn.Module):
    def __init__(self, channel):
        super(replace_aggregation_final, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        # self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        # self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        # self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        # self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        #
        # self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        # self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        # self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):

        x1 = self.upsample(x1)
        x1 = self.upsample(x1)
        x2 = self.upsample(x2)
        x = torch.cat((x1, x2, x3), 1)

        # x1_1 = x1
        # x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        # x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
        #        * self.conv_upsample3(self.upsample(x2)) * x3
        #
        # x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        # x2_2 = self.conv_concat2(x2_2)
        #
        # x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)

        x3_2 = self.conv_concat3(x)

        return x3_2


class SemiModel_without_agg_final(nn.Module):
    def __init__(self, channel=32):
        super(SemiModel_without_agg_final, self).__init__()

        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512

        self.rfb3 = GCM(256, channel)
        self.rfb4 = GCM(512, channel)
        self.rfb5 = GCM(512, channel)
        self.agg1 = aggregation_init(channel)

        self.rfb1 = GCM(64, channel)
        self.rfb2 = GCM(128, channel)
        self.rfb3_2 = GCM(channel, channel)
        # self.agg2 = aggregation_final(channel)
        self.agg2 = replace_aggregation_final(channel)


        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.HA = Refine()

        self.atten_A_channel_1 = ChannelAttention(64)
        self.atten_A_channel_2 = ChannelAttention(128)
        self.atten_A_channel_3 = ChannelAttention(256)
        self.atten_A_channel_4 = ChannelAttention(512)
        self.atten_A_channel_5 = ChannelAttention(512)

        self.atten_A_spatial_1 = SpatialAttention()
        self.atten_A_spatial_2 = SpatialAttention()
        self.atten_A_spatial_3 = SpatialAttention()
        self.atten_A_spatial_4 = SpatialAttention()
        self.atten_A_spatial_5 = SpatialAttention()

        self.atten_B_channel_1 = ChannelAttention(64)
        self.atten_B_channel_2 = ChannelAttention(128)
        self.atten_B_channel_3 = ChannelAttention(256)
        self.atten_B_channel_4 = ChannelAttention(512)
        self.atten_B_channel_5 = ChannelAttention(512)

        self.atten_B_spatial_1 = SpatialAttention()
        self.atten_B_spatial_2 = SpatialAttention()
        self.atten_B_spatial_3 = SpatialAttention()
        self.atten_B_spatial_4 = SpatialAttention()
        self.atten_B_spatial_5 = SpatialAttention()

        self.agant1 = self._make_agant_layer(32 * 3, 32 * 2)
        self.agant2 = self._make_agant_layer(32 * 2, 32)
        self.out_conv = nn.Conv2d(32 * 1, 1, kernel_size=1, stride=1, bias=True)

    def forward(self, A, B):
        layer1_A = self.inc(A)
        layer2_A = self.down1(layer1_A)
        layer3_A = self.down2(layer2_A)
        layer4_A = self.down3(layer3_A)
        layer5_A = self.down4(layer4_A)

        layer1_B = self.inc(B)
        layer2_B = self.down1(layer1_B)
        layer3_B = self.down2(layer2_B)
        layer4_B = self.down3(layer3_B)
        layer5_B = self.down4(layer4_B)

        layer1_A = layer1_A.mul(self.atten_A_channel_1(layer1_A))
        layer1_A = layer1_A.mul(self.atten_A_spatial_1(layer1_A))

        layer1_B = layer1_B.mul(self.atten_B_channel_1(layer1_B))
        layer1_B = layer1_B.mul(self.atten_B_spatial_1(layer1_B))
        layer1 = layer1_A + layer1_B

        layer2_A = layer2_A.mul(self.atten_A_channel_2(layer2_A))
        layer2_A = layer2_A.mul(self.atten_A_spatial_2(layer2_A))

        layer2_B = layer2_B.mul(self.atten_B_channel_2(layer2_B))
        layer2_B = layer2_B.mul(self.atten_B_spatial_2(layer2_B))
        layer2 = layer2_A + layer2_B

        layer3_A = layer3_A.mul(self.atten_A_channel_3(layer3_A))
        layer3_A = layer3_A.mul(self.atten_A_spatial_3(layer3_A))

        layer3_B = layer3_B.mul(self.atten_B_channel_3(layer3_B))
        layer3_B = layer3_B.mul(self.atten_B_spatial_3(layer3_B))
        layer3 = layer3_A + layer3_B

        layer4_A = layer4_A.mul(self.atten_A_channel_4(layer4_A))
        layer4_A = layer4_A.mul(self.atten_A_spatial_4(layer4_A))

        layer4_B = layer4_B.mul(self.atten_B_channel_4(layer4_B))
        layer4_B = layer4_B.mul(self.atten_B_spatial_4(layer4_B))
        layer4 = layer4_A + layer4_B

        layer5_A = layer5_A.mul(self.atten_A_channel_5(layer5_A))
        layer5_A = layer5_A.mul(self.atten_A_spatial_5(layer5_A))

        layer5_B = layer5_B.mul(self.atten_B_channel_5(layer5_B))
        layer5_B = layer5_B.mul(self.atten_B_spatial_5(layer5_B))
        layer5 = layer5_A + layer5_B

        layer3 = self.rfb3(layer3)
        layer4 = self.rfb4(layer4)
        layer5 = self.rfb5(layer5)
        attention_map = self.agg1(layer5, layer4, layer3)

        layer1, layer2, layer3 = self.HA(attention_map.sigmoid(), layer1, layer2, layer3)

        layer1 = self.rfb1(layer1)
        layer2 = self.rfb2(layer2)
        layer3 = self.rfb3_2(layer3)

        y = self.agg2(layer3, layer2, layer1)  # *4

        y = self.agant1(y)
        y = self.agant2(y)
        y = self.out_conv(y)

        return F.interpolate(attention_map, size=A.size()[2:], mode='bilinear', align_corners=True), F.interpolate(y,
                                                                                                                   size=A.size()[
                                                                                                                        2:],
                                                                                                                   mode='bilinear',
                                                                                                                   align_corners=True)

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

