# !/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class RPN_net(nn.Module):
    def __init__(self, channel, num_anchor):
        super(RPN_net, self).__init__()

        self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(channel, num_anchor * 2, 1, padding=0)
        self.conv3 = nn.Conv2d(channel, num_anchor * 4, 1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, C):
        rpn_logits = []
        rpn_loc = []
        for i in range(5):
            x = self.conv1(C[i])
            x = self.relu(x)

            rpn_logits.append(self.conv2(x).permute(0, 2, 3, 1).contiguous().view(-1, 2))
            rpn_loc.append(self.conv3(x).permute(0, 2, 3, 1).contiguous().view(-1, 4))

        return torch.cat(rpn_logits, dim=0), torch.cat(rpn_loc, dim=0)
