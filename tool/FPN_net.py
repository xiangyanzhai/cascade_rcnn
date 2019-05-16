# !/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class FPN_net(nn.Module):
    def __init__(self, channel):
        super(FPN_net, self).__init__()

        self.conv5_1x1 = nn.Conv2d(2048, channel, 1)
        self.conv4_1x1 = nn.Conv2d(1024, channel, 1)
        self.conv3_1x1 = nn.Conv2d(512, channel, 1)
        self.conv2_1x1 = nn.Conv2d(256, channel, 1)
        self.conv5_3x3 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv4_3x3 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv3_3x3 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv2_3x3 = nn.Conv2d(channel, channel, 3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         # nn.init.xavier_uniform_(m.weight)
        #         nn.init.kaiming_uniform_(m.weight,a=1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, C):
        C2, C3, C4, C5 = C
        P5 = self.conv5_1x1(C5)
        P4 = self.conv4_1x1(C4) + F.interpolate(P5, size=C4.shape[2:])
        P3 = self.conv3_1x1(C3) + F.interpolate(P4, size=C3.shape[2:])
        P2 = self.conv2_1x1(C2) + F.interpolate(P3, size=C2.shape[2:])

        P5 = self.conv5_3x3(P5)
        P4 = self.conv4_3x3(P4)
        P3 = self.conv3_3x3(P3)
        P2 = self.conv2_3x3(P2)
        P6 = self.maxpool(P5)



        return P2, P3, P4, P5, P6
        pass
