# !/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn


class Mask_net(nn.Module):
    def __init__(self,channel, num_cls):
        super(Mask_net, self).__init__()
        self.mask_net = nn.Sequential(
            nn.Conv2d(channel,channel, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(channel, channel, 2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(channel, num_cls + 1, 1, padding=0)
        )
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         # Caffe2 implementation uses MSRAFill, which in fact
        #         # corresponds to kaiming_normal_ in PyTorch
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mask_net(x)
        return x
