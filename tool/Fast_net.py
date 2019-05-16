# !/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn


class Fast_net(nn.Module):
    def __init__(self, num_classes, channel_in, channel):
        super(Fast_net, self).__init__()
        self.num_classes = num_classes
        self.fast_head = nn.Sequential(
            nn.Linear(channel_in, channel),
            nn.ReLU(True),
            nn.Linear(channel, channel),
            nn.ReLU(True),
        )
        self.Linear1 = nn.Linear(channel, num_classes + 1)
        self.Linear2 = nn.Linear(channel, (num_classes + 1) * 4)
        nn.init.normal_(self.Linear1.weight, std=0.01)
        nn.init.normal_(self.Linear2.weight, std=0.001)
        nn.init.constant_(self.Linear1.bias, 0)
        nn.init.constant_(self.Linear2.bias, 0)
        pass

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fast_head(x)
        fast_logits = self.Linear1(x)
        fast_loc = self.Linear2(x)
        fast_loc = fast_loc.view(fast_loc.shape[0], -1, 4)
        return fast_logits, fast_loc
        pass
