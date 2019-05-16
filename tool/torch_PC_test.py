# !/usr/bin/python
# -*- coding:utf-8 -*-
import torch
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.layers import nms as _box_nms


# from chainer.backends import cuda
# from chainercv.utils.bbox.non_maximum_suppression import \
#     non_maximum_suppression


def loc2bbox(pre_loc, anchor):
    c_hw = anchor[..., 2:4] - anchor[..., 0:2]
    c_yx = anchor[..., :2] + c_hw / 2
    yx = pre_loc[..., :2] * c_hw + c_yx
    hw = torch.exp(pre_loc[..., 2:4]) * c_hw
    yx1 = yx - hw / 2
    yx2 = yx + hw / 2
    bboxes = torch.cat((yx1, yx2), dim=-1)
    return bboxes


class ProposalCreator(object):
    def __init__(self,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 force_cpu_nms=False,
                 min_size=16,
                 ):

        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.force_cpu_nms = force_cpu_nms
        self.min_size = min_size

    def __call__(self, loc, score,
                 anchor, img_size, train=True, scale=1.):
        if train:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        # print('======================', n_pre_nms, n_post_nms, self.min_size, self.nms_thresh,
        #       '==============================')
        h, w = img_size

        roi = loc2bbox(loc, anchor)
        roi[:, slice(0, 4, 2)] = torch.clamp(roi[:, slice(0, 4, 2)], 0, w)
        roi[:, slice(1, 4, 2)] = torch.clamp(roi[:, slice(1, 4, 2)], 0, h)
        hw = roi[:, 2:4] - roi[:, :2]
        inds = hw >= self.min_size
        inds = inds.all(dim=1)
        roi = roi[inds]
        score = score[inds]

        score, inds = score.sort(descending=True)
        score = score[:n_pre_nms]
        inds = inds[:n_pre_nms]
        roi = roi[inds]
        inds = _box_nms(roi, score, self.nms_thresh)[:n_post_nms]
        # inds = _box_nms(roi.cpu().cuda(), score.cpu().cuda(), self.nms_thresh)[:n_post_nms]

        roi = roi[inds]

        return roi
