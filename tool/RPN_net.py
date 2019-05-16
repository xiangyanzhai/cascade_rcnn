# !/usr/bin/python
# -*- coding:utf-8 -*-
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        rpn_logits = self.conv2(x)
        rpn_loc = self.conv3(x)
        rpn_logits = rpn_logits.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        rpn_loc = rpn_loc.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        return rpn_logits, rpn_loc
