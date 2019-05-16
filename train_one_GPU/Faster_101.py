# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import torch.nn as nn
is_gpu = torch.cuda.is_available()


def cuda(x):
    if is_gpu:
        x = x.cuda()
    return x

# torch.backends.cudnn.benchmark =True
from cascade_rcnn.tool.config import Config
import torch.utils.model_zoo as model_zoo
from cascade_rcnn.tool.get_anchors import get_anchors
from cascade_rcnn.tool.torch_ATC_test import AnchorTargetCreator
from cascade_rcnn.tool.torch_PC_test import ProposalCreator
from cascade_rcnn.tool.torch_PTC_test import ProposalTargetCreator
from maskrcnn_benchmark.layers import ROIAlign
from cascade_rcnn.tool.RPN_net import RPN_net
import torch.nn.functional as F
from cascade_rcnn.tool.read_Data import Read_Data
from torch.utils.data import DataLoader

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Fast_net(nn.Module):
    def __init__(self, num_classes):
        super(Fast_net, self).__init__()
        self.Linear1 = nn.Linear(2048, num_classes + 1)
        self.Linear2 = nn.Linear(2048, (num_classes + 1) * 4)
        nn.init.normal_(self.Linear1.weight, std=0.01)
        nn.init.normal_(self.Linear2.weight, std=0.001)
        pass

    def forward(self, x):
        fast_logits = self.Linear1(x)
        fast_loc = self.Linear2(x)
        fast_loc = fast_loc.view(fast_loc.shape[0], -1, 4)
        return fast_logits, fast_loc


ce_loss = nn.CrossEntropyLoss()
roialign = ROIAlign((14, 14), 1 / 16., 2)


def SmoothL1Loss(net_loc_train, loc, sigma, num):
    t = torch.abs(net_loc_train - loc)
    a = t[t < 1]
    b = t[t >= 1]
    loss1 = (a * sigma) ** 2 / 2
    loss2 = b - 0.5 / sigma ** 2
    loss = (loss1.sum() + loss2.sum()) / num
    return loss


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, config, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.config = config
        self.Mean = torch.tensor(config.Mean, dtype=torch.float32)
        self.num_anchor = len(config.anchor_scales) * len(config.anchor_ratios)
        self.anchors = get_anchors(np.ceil(self.config.img_max / 16 + 1), self.config.anchor_scales,
                                   self.config.anchor_ratios)
        self.ATC = AnchorTargetCreator(n_sample=config.rpn_n_sample, pos_iou_thresh=config.rpn_pos_iou_thresh,
                                       neg_iou_thresh=config.rpn_neg_iou_thresh, pos_ratio=config.rpn_pos_ratio)
        self.PC = ProposalCreator(nms_thresh=config.roi_nms_thresh,
                                  n_train_pre_nms=config.roi_train_pre_nms,
                                  n_train_post_nms=config.roi_train_post_nms,
                                  n_test_pre_nms=config.roi_test_pre_nms, n_test_post_nms=config.roi_test_post_nms,
                                  min_size=config.roi_min_size)
        self.PTC = ProposalTargetCreator(n_sample=config.fast_n_sample,
                                         pos_ratio=config.fast_pos_ratio, pos_iou_thresh=config.fast_pos_iou_thresh,
                                         neg_iou_thresh_hi=config.fast_neg_iou_thresh_hi,
                                         neg_iou_thresh_lo=config.fast_neg_iou_thresh_lo)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        for i in range(2, 5):
            print('layer%d' % i)
            getattr(self, 'layer%d' % i)[0].conv1.stride = (2, 2)
            getattr(self, 'layer%d' % i)[0].conv2.stride = (1, 1)

        self.rpn = RPN_net(1024, self.num_anchor)
        self.fast = Fast_net(config.num_cls)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def rpn_loss(self, rpn_logits, rpn_loc, bboxes, tanchors, img_size):
        inds, label, indsP, loc = self.ATC(bboxes, tanchors, img_size)

        rpn_logits_train = rpn_logits[inds]
        rpn_loc_train = rpn_loc[indsP]
        rpn_cls_loss = ce_loss(rpn_logits_train, label)
        rpn_box_loss = SmoothL1Loss(rpn_loc_train, loc, 3.0, 240.0)
        self.a = rpn_cls_loss
        self.b = rpn_box_loss
        return rpn_cls_loss, rpn_box_loss

    def fast_train_data(self, loc, score, anchor, img_size, bboxes):
        roi = self.PC(loc, score, anchor, img_size)
        roi, loc, label = self.PTC(roi, bboxes[:, :4], bboxes[:, -1].long())
        roi_inds = cuda(torch.zeros((roi.size()[0], 1)))

        roi = torch.cat([roi_inds, roi], dim=1)
        return roi, loc, label

    def fast_loss(self, fast_logits, fast_loc, label, loc):
        fast_num = label.shape[0]
        fast_num_P = loc.shape[0]
        fast_loc_train = fast_loc[torch.arange(fast_num_P), label[:fast_num_P].long()]

        fast_cls_loss = ce_loss(fast_logits, label.long())
        fast_box_loss = SmoothL1Loss(fast_loc_train, loc, 1.0, float(fast_num))
        self.c = fast_cls_loss
        self.d = fast_box_loss
        self.fast_num = fast_num
        self.fast_num_P = fast_num_P
        return fast_cls_loss, fast_box_loss

    def process_im(self, x, bboxes):
        x = x[None]
        x = x[..., [2, 1, 0]]
        x = x.permute(0, 3, 1, 2)
        H, W = x.shape[2:]
        ma = max(H, W)
        mi = min(H, W)
        scale = min(self.config.img_max / ma, self.config.img_min / mi)
        nh = int(H * scale)
        nw = int(W * scale)
        x = F.interpolate(x, size=(nh, nw))
        bboxes[:, :4] = bboxes[:, :4] * scale
        x = x.permute(0, 2, 3, 1)

        # NHWC RGB
        return x, bboxes

    def get_loss(self, x, bboxes, num_b, H, W):
        x = x.view(-1)[:H * W * 3].view(H, W, 3)
        bboxes = bboxes[:num_b]

        inds = bboxes[:, -1] >= 0
        bboxes = bboxes[inds]

        x = cuda(x.float())
        bboxes = cuda(bboxes)
        x, bboxes = self.process_im(x, bboxes)
        x = x - cuda(self.Mean)
        x = x[..., [2, 1, 0]]
        x = x.permute(0, 3, 1, 2)
        img_size = x.shape[2:]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        rpn_logits, rpn_loc = self.rpn(x)
        map_H, map_W = x.shape[2:]
        tanchors = self.anchors[:map_H, :map_W]
        tanchors = cuda(tanchors.contiguous().view(-1, 4))

        rpn_cls_loss, rpn_box_loss = self.rpn_loss(rpn_logits, rpn_loc, bboxes, tanchors, img_size)
        roi, loc, label = self.fast_train_data(rpn_loc.data, F.softmax(rpn_logits.data, dim=-1)[:, 1], tanchors,
                                               img_size, bboxes)

        x = roialign(x, roi)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        fast_logits, fast_loc = self.fast(x)
        fast_cls_loss, fast_box_loss = self.fast_loss(fast_logits, fast_loc, label, loc)
        return rpn_cls_loss + rpn_box_loss + fast_cls_loss + fast_box_loss

    def forward(self, imgs, bboxes, num_b, num_H, num_W):
        loss = list(map(self.get_loss, imgs, bboxes, num_b, num_H, num_W))
        return sum(loss)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(config, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(config, Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def func(batch):
    m = len(batch)
    num_b = []
    num_H = []
    num_W = []
    for i in range(m):
        num_b.append(batch[i][2])
        num_H.append(batch[i][3])
        num_W.append(batch[i][4])

    max_b = max(num_b)
    max_H = max(num_H)
    max_W = max(num_W)
    imgs = []
    bboxes = []
    for i in range(m):
        imgs.append(batch[i][0].resize_(max_H, max_W, 3)[None])
        bboxes.append(batch[i][1].resize_(max_b, 5)[None])

    imgs = torch.cat(imgs, dim=0)
    bboxes = torch.cat(bboxes, dim=0)
    return imgs, bboxes, torch.tensor(num_b, dtype=torch.int64), torch.tensor(num_H, dtype=torch.int64), torch.tensor(
        num_W, dtype=torch.int64)


from datetime import datetime


def train(model, config, step, x, pre_model_file, model_file=None):
    model = model(config)
    print(model)
    model.eval()
    model_dic = model.state_dict()
    pretrained_dict = torch.load(pre_model_file, map_location='cpu')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dic}
    print(len(pretrained_dict))
    print('*******', len(pretrained_dict))
    model_dic.update(pretrained_dict)
    model.load_state_dict(model_dic)
    if step > 0:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        print(model_file)
    else:
        print(pre_model_file)
    cuda(model)
    train_params = list(model.parameters())

    lr = config.lr * config.batch_size_per_GPU
    if step >= 60000 * x:
        lr = lr / 10
    if step >= 80000 * x:
        lr = lr / 10
    print('lr        ******************', lr)
    print('weight_decay     ******************', config.weight_decay)

    if False:
        bias_p = []
        weight_p = []
        print(len(train_params))
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        print(len(weight_p), len(bias_p))
        opt = torch.optim.SGD(
            [{'params': weight_p, 'weight_decay': config.weight_decay, 'lr': lr},
             {'params': bias_p, 'lr': lr * config.bias_lr_factor}],
            momentum=0.9, )
    else:
        bias_p = []
        weight_p = []
        bn_weight_p = []
        print(len(train_params))
        for name, p in model.named_parameters():
            print(name, p.shape)
            if len(p.shape) == 1:
                if 'bias' in name:
                    bias_p.append(p)
                else:
                    bn_weight_p.append(p)
            else:
                weight_p.append(p)
        print(len(weight_p), len(bias_p), len(bn_weight_p))
        opt = torch.optim.SGD([{'params': weight_p, 'weight_decay': config.weight_decay, 'lr': lr},
                               {'params': bn_weight_p, 'lr': lr},
                               {'params': bias_p, 'lr': lr * config.bias_lr_factor}],
                              momentum=0.9, )
    dataset = Read_Data(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size_per_GPU, collate_fn=func,
                            shuffle=True, drop_last=True, pin_memory=True, num_workers=16)
    epochs = 10000
    flag = False
    print('start:  step=', step)
    for epoch in range(epochs):
        for imgs, bboxes, num_b, num_H, num_W in dataloader:
            loss = model(imgs, bboxes, num_b, num_H, num_W)
            loss = loss / imgs.shape[0]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 10, norm_type=2)
            opt.step()
            if step % 20 == 0:
                print(datetime.now(), 'loss:%.4f' % loss, 'rpn_cls_loss:%.4f' % model.a,
                      'rpn_box_loss:%.4f' % model.b,
                      'fast_cls_loss:%.4f' % model.c, 'fast_box_loss:%.4f' % model.d,
                      model.fast_num,
                      model.fast_num_P, step)
            step += 1
            if step == int(60000 * x) or step == int(80000 * x):

                for param_group in opt.param_groups:
                    param_group['lr'] = param_group['lr'] / 10
                    print('*****************************************************************', param_group['lr'])

            if (step <= 10000 and step % 1000 == 0) or step % 5000 == 0 or step == 1:
                torch.save(model.state_dict(), './models/Faster_101_%d_1.pth' % step)
            #     # faster.save('./model/vgg16_%d.pth'%c)
            if step >= 90010 * x:
                flag = True
                break
        if flag:
            break
    torch.save(model.state_dict(), './models/Faster_101_final_1.pth')


if __name__ == "__main__":
    Mean = [123.68, 116.78, 103.94]
    path = '/home/zhai/PycharmProjects/Demo35/cascade_rcnn/data_preprocess/'
    Bboxes = [path + 'Bboxes_07.pkl', path + 'Bboxes_12.pkl']
    img_paths = [path + 'img_paths_07.pkl', path + 'img_paths_12.pkl']

    files = [img_paths, Bboxes]
    config = Config(True, Mean, files, lr=0.001, weight_decay=0.0001, batch_size_per_GPU=1, img_max=1000, img_min=600,
                    roi_min_size=16,
                    bias_lr_factor=2)

    step = 0
    model = resnet101
    x = 1
    pre_model_file = '/home/zhai/PycharmProjects/Demo35/pytorch_Faster_tool/resnet_caffe/resnet101-caffe.pth'
    model_file = ''
    train(model, config, step, x, pre_model_file, model_file=model_file)
