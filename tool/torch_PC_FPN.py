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
                 min_size=[4, 8, 16, 32, 64],
                 ):

        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.force_cpu_nms = force_cpu_nms
        self.min_size = min_size

    def __call__(self, loc, score,
                 anchor, img_size, map_HW, train=True, scale=1.):
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

        Roi = []
        nms_Roi = []
        Score = []
        C = 0


        for i in range(5):
            map_H, map_W = map_HW[i]

            c = map_H * map_W * 3
            tscore = score[C:C + c]
            troi = roi[C:C + c]
            C += c
            hw = troi[:, 2:4] - troi[:, :2]
            inds = hw >= self.min_size[i]
            inds = inds.all(dim=1)
            troi = troi[inds]
            tscore = tscore[inds]
            tscore, inds = tscore.sort(descending=True)
            inds = inds[:n_post_nms]
            troi = troi[inds]
            tscore = tscore[:n_post_nms]
            Roi.append(troi)
            Score.append(tscore)
            nms_Roi.append(troi + i * 2 * max(h, w))

        roi = torch.cat(Roi, dim=0)
        nms_roi = torch.cat(nms_Roi, dim=0)
        score = torch.cat(Score, dim=0)
        score, inds = score.sort(descending=True)
        roi = roi[inds]
        inds = _box_nms(nms_roi, score, self.nms_thresh)[:n_post_nms]

        roi = roi[inds]

        return roi
