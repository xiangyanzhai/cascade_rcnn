# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os
import numpy as np

import torch

is_gpu = torch.cuda.is_available()


def cuda(x):
    if is_gpu:
        x = x.cuda()
    return x


import torch.nn as nn
# torch.backends.cudnn.benchmark =True
from cascade_rcnn.tool.config import Config
from cascade_rcnn.tool.get_anchors import get_anchors
from cascade_rcnn.tool.torch_PC_test import ProposalCreator
from maskrcnn_benchmark.layers import ROIAlign
from torchvision.models import vgg16
from cascade_rcnn.tool.RPN_net import RPN_net
from cascade_rcnn.tool.Fast_net import Fast_net
import torch.nn.functional as F
from cascade_rcnn.tool.cascade_predict import predict

roialign = ROIAlign((7, 7), 1 / 16., 2)


def SmoothL1Loss(net_loc_train, loc, sigma, num):
    t = torch.abs(net_loc_train - loc)
    a = t[t < 1]
    b = t[t >= 1]
    loss1 = (a * sigma) ** 2 / 2
    loss2 = b - 0.5 / sigma ** 2
    loss = (loss1.sum() + loss2.sum()) / num
    return loss


class Faster_Rcnn(nn.Module):
    def __init__(self, config):
        super(Faster_Rcnn, self).__init__()
        self.config = config
        self.Mean = torch.tensor(config.Mean, dtype=torch.float32)
        self.num_anchor = len(config.anchor_scales) * len(config.anchor_ratios)
        self.anchors = get_anchors(np.ceil(self.config.img_max / 16 + 1), self.config.anchor_scales,
                                   self.config.anchor_ratios)

        self.PC = ProposalCreator(nms_thresh=config.roi_nms_thresh,
                                  n_train_pre_nms=config.roi_train_pre_nms,
                                  n_train_post_nms=config.roi_train_post_nms,
                                  n_test_pre_nms=config.roi_test_pre_nms, n_test_post_nms=config.roi_test_post_nms,
                                  min_size=config.roi_min_size)

        self.features = vgg16().features[:-1]
        self.rpn = RPN_net(512, self.num_anchor)
        self.fast_1 = Fast_net(config.num_cls, 512 * 7 * 7, 2048)
        self.fast_2 = Fast_net(config.num_cls, 512 * 7 * 7, 2048)
        self.fast_3 = Fast_net(config.num_cls, 512 * 7 * 7, 2048)

        self.loc_std1 = [1. / 10, 1. / 10, 1. / 5, 1. / 5]
        self.loc_std2 = [1. / 20, 1. / 20, 1. / 10, 1. / 10]
        self.loc_std3 = [1. / 30, 1. / 30, 1. / 15, 1. / 15]
        self.weights = [1.0, 1.0, 1.0]

    def process_im(self, x):
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
        x = x.permute(0, 2, 3, 1)
        return x, scale

    def roi_layer(self, roi):
        roi_inds = cuda(torch.zeros((roi.size()[0], 1)))
        roi = torch.cat([roi_inds, roi], dim=1)
        return roi

    def forward(self, x):
        x = x.float()
        x = cuda(x)

        x, scale = self.process_im(x)
        x = x - cuda(self.Mean)
        x = x.permute(0, 3, 1, 2)

        img_size = x.shape[2:]

        x = self.features(x)
        rpn_logits, rpn_loc = self.rpn(x)
        map_H, map_W = x.shape[2:]
        tanchors = self.anchors[:map_H, :map_W]
        tanchors = tanchors.contiguous().view(-1, 4)
        tanchors = cuda(tanchors)

        roi = self.PC(rpn_loc, F.softmax(rpn_logits, dim=-1)[:, 1], tanchors, img_size, train=self.config.is_train)
        roi = self.roi_layer(roi)
        xx = roialign(x, roi)
        fast_logits, fast_loc = self.fast_1(xx)

        fast_loc = fast_loc[:, 1:] * cuda(torch.tensor(self.loc_std1))
        roi = self.loc2bbox(fast_loc, roi[:, 1:][:, None])
        score = F.softmax(fast_logits, dim=-1)[:, 1:]
        pre_bboxes = roi * self.weights[0]
        pre_score = score * self.weights[0]
        _, inds = score.max(dim=-1)
        t = torch.arange(score.shape[0])
        roi = roi[t, inds]
        roi, inds = self.filter_bboxes(roi, img_size, self.config.roi_min_size)
        pre_bboxes = pre_bboxes[inds]
        pre_score = pre_score[inds]

        roi = self.roi_layer(roi)
        xx = roialign(x, roi)
        fast_logits, fast_loc = self.fast_2(xx)

        fast_loc = fast_loc[:, 1:] * cuda(torch.tensor(self.loc_std2))
        roi = self.loc2bbox(fast_loc, roi[:, 1:][:, None])
        score = F.softmax(fast_logits, dim=-1)[:, 1:]

        pre_bboxes = pre_bboxes + roi * self.weights[1]
        pre_score = pre_score + score * self.weights[1]
        _, inds = score.max(dim=-1)
        t = torch.arange(score.shape[0])
        roi = roi[t, inds]
        roi, inds = self.filter_bboxes(roi, img_size, self.config.roi_min_size)
        pre_bboxes = pre_bboxes[inds]
        pre_score = pre_score[inds]

        roi = self.roi_layer(roi)
        xx = roialign(x, roi)
        fast_logits, fast_loc = self.fast_3(xx)
        fast_loc = fast_loc[:, 1:] * cuda(torch.tensor(self.loc_std3))
        roi = self.loc2bbox(fast_loc, roi[:, 1:][:, None])
        score = F.softmax(fast_logits, dim=-1)[:, 1:]
        pre_bboxes = pre_bboxes + roi * self.weights[2]
        pre_score = pre_score + score * self.weights[2]
        pre_bboxes = pre_bboxes / sum(self.weights)
        pre_score = pre_score / sum(self.weights)

        pre = predict(pre_bboxes, pre_score, img_size[0], img_size[1], iou_thresh_=0.5, c_thresh=0.05)[:100]
        pre[:, :4] = pre[:, :4] / scale
        return pre

    def filter_bboxes(self, roi, img_size, roi_min_size):
        h, w = img_size
        roi[:, slice(0, 4, 2)] = torch.clamp(roi[:, slice(0, 4, 2)], 0, w)
        roi[:, slice(1, 4, 2)] = torch.clamp(roi[:, slice(1, 4, 2)], 0, h)
        hw = roi[:, 2:4] - roi[:, :2]
        inds = hw >= roi_min_size
        inds = inds.all(dim=-1)
        roi = roi[inds]
        return roi, inds

    def loc2bbox(self, pre_loc, anchor):
        c_hw = anchor[..., 2:4] - anchor[..., 0:2]
        c_yx = anchor[..., :2] + c_hw / 2
        yx = pre_loc[..., :2] * c_hw + c_yx
        hw = torch.exp(pre_loc[..., 2:4]) * c_hw
        yx1 = yx - hw / 2
        yx2 = yx + hw / 2
        bboxes = torch.cat((yx1, yx2), dim=-1)
        return bboxes


from datetime import datetime
from cascade_rcnn.mAP.voc_mAP import mAP
import cv2
from sklearn.externals import joblib


def test(model, config, model_file):
    model = model(config)
    model.eval()
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    cuda(model)

    test_dir = r'/home/zhai/PycharmProjects/Demo35/dataset/voc/VOCtest2007/VOCdevkit/VOC2007/JPEGImages/'
    names = os.listdir(test_dir)
    names = [name.split('.')[0] for name in names]

    names = sorted(names)

    i = 0
    m = 100000000

    Res = {}
    start_time = datetime.now()

    for name in names[:m]:
        i += 1
        print(datetime.now(), i)
        im_file = test_dir + name + '.jpg'
        img = cv2.imread(im_file)
        img = torch.tensor(img)
        res = model(img)
        res = res.cpu()
        res = res.detach().numpy()

        Res[name] = res

    print('==========', datetime.now() - start_time)
    joblib.dump(Res, 'Faster_vgg16_cascade_1.pkl')
    GT = joblib.load('../mAP/voc_GT.pkl')
    AP = mAP(Res, GT, 20, iou_thresh=0.5, use_07_metric=True, e=0.01)
    print(AP)
    AP = AP.mean()
    print(AP)


if __name__ == "__main__":
    Mean = [123.68, 116.78, 103.94]
    model = Faster_Rcnn
    config = Config(False, Mean, None, img_max=1000, img_min=600, roi_min_size=16)
    model_file = '/home/zhai/PycharmProjects/Demo35/cascade_rcnn/train_one_GPU/models/vgg16_cascade_90000_1.pth'
    test(model, config, model_file)
