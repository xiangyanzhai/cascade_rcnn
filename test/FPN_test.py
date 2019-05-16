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

from cascade_rcnn.tool.get_anchors import get_anchors

from cascade_rcnn.tool.torch_PC_FPN import ProposalCreator

from maskrcnn_benchmark.layers import ROIAlign
from cascade_rcnn.tool.resnet import resnet101
from cascade_rcnn.tool.FPN_net import FPN_net
from cascade_rcnn.tool.FPN_RPN import RPN_net
from cascade_rcnn.tool.FPN_Fast import Fast_net
from cascade_rcnn.tool.faster_predict import predict
import torch.nn.functional as F

roialign_list = [ROIAlign((7, 7), 1 / 4., 2), ROIAlign((7, 7), 1 / 8., 2), ROIAlign((7, 7), 1 / 16., 2),
                 ROIAlign((7, 7), 1 / 32., 2)]


class Faster_Rcnn(nn.Module):
    def __init__(self, config):
        super(Faster_Rcnn, self).__init__()
        self.config = config
        self.Mean = torch.tensor(config.Mean, dtype=torch.float32)
        self.num_anchor = len(config.anchor_scales) * len(config.anchor_ratios)
        self.anchors = []
        self.num_anchor = []
        for i in range(5):
            self.num_anchor.append(len(config.anchor_scales[i]) * len(config.anchor_ratios[i]))
            stride = 4 * 2 ** i
            print(stride, self.config.anchor_scales[i], self.config.anchor_ratios[i])
            anchors = get_anchors(np.ceil(self.config.img_max / stride + 1), self.config.anchor_scales[i],
                                  self.config.anchor_ratios[i], stride=stride)
            print(anchors.shape)
            self.anchors.append(anchors)

        self.PC = ProposalCreator(nms_thresh=config.roi_nms_thresh,
                                  n_train_pre_nms=config.roi_train_pre_nms,
                                  n_train_post_nms=config.roi_train_post_nms,
                                  n_test_pre_nms=config.roi_test_pre_nms, n_test_post_nms=config.roi_test_post_nms,
                                  min_size=config.roi_min_size)

        self.features = resnet101()
        self.fpn = FPN_net(256)
        self.rpn = RPN_net(256, self.num_anchor[0])
        self.fast = Fast_net(config.num_cls, 256 * 7 * 7, 1024)
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.fast_num = 0
        self.fast_num_P = 0

    def roi_layer(self, loc, score, anchor, img_size, map_HW):
        roi = self.PC(loc, score, anchor, img_size, map_HW, train=self.config.is_train)
        area = roi[:, 2:] - roi[:, :2] + 1
        area = area.prod(dim=-1)
        roi_inds = torch.floor(4.0 + torch.log(area ** 0.5 / 224.0) / np.log(2.0))
        roi_inds = roi_inds.clamp(2, 5) - 2

        roi = torch.cat([cuda(torch.zeros(roi.shape[0], 1)), roi], dim=-1)
        return roi, roi_inds

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

        # NHWC RGB
        return x, scale

    def pooling(self, P, roi, roi_inds):
        x = []
        inds = []
        index = cuda(torch.arange(roi.shape[0]))
        for i in range(4):
            t = roi_inds == i
            x.append(roialign_list[i](P[i], roi[t]))
            inds.append(index[t])
        x = torch.cat(x, dim=0)
        inds = torch.cat(inds, dim=0)
        inds = inds.argsort()
        x = x[inds]
        return x

    def forward(self, x):
        x = x.float()
        x = cuda(x)
        x, scale = self.process_im(x)
        x = x - cuda(self.Mean)
        x = x[..., [2, 1, 0]]
        x = x.permute(0, 3, 1, 2)
        img_size = x.shape[2:]
        C = self.features(x)
        P = self.fpn(C)
        rpn_logits, rpn_loc = self.rpn(P)
        tanchors = []
        map_HW = []
        for i in range(5):
            H, W = P[i].shape[2:4]
            map_HW.append((H, W))
            tanchors.append(self.anchors[i][:H, :W].contiguous().view(-1, 4))
        tanchors = cuda(torch.cat(tanchors, dim=0))

        roi, roi_inds = self.roi_layer(rpn_loc.data, F.softmax(rpn_logits.data, dim=-1)[:, 1],
                                       tanchors,
                                       img_size, map_HW)

        x = self.pooling(P, roi, roi_inds)
        fast_logits, fast_loc = self.fast(x)
        fast_loc = fast_loc * cuda(torch.tensor([0.1, 0.1, 0.2, 0.2]))
        pre = predict(fast_loc, F.softmax(fast_logits, dim=-1), roi[:, 1:], img_size[0], img_size[1], )
        pre[:, :4] = pre[:, :4] / scale
        return pre


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
    m = 100

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
    joblib.dump(Res, 'FPN_101.pkl')
    GT = joblib.load('../mAP/voc_GT.pkl')
    AP = mAP(Res, GT, 20, iou_thresh=0.5, use_07_metric=True, e=0.05)
    print(AP)
    AP = AP.mean()
    print(AP)


if __name__ == "__main__":
    Mean = [123.68, 116.78, 103.94]

    config = Config(False, Mean, None, lr=0.00125, weight_decay=0.0001, img_max=1333, img_min=800,
                    anchor_scales=[[32], [64], [128], [256], [512]],
                    anchor_ratios=[[0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2]], fast_n_sample=512,
                    roi_min_size=[4, 8, 16, 32, 64],
                    roi_train_pre_nms=12000,
                    roi_train_post_nms=2000,
                    roi_test_pre_nms=6000,
                    roi_test_post_nms=1000, )
    model = Faster_Rcnn
    model_file = '/home/zhai/PycharmProjects/Demo35/cascade_rcnn/train_one_GPU/models/FPN_101_90000_1.pth'

    test(model, config, model_file)
