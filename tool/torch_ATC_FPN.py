# !/usr/bin/python
# -*- coding:utf-8 -*-
import torch

is_gpu = torch.cuda.is_available()


def cuda(x):
    if is_gpu:
        x = x.cuda()
    return x


def cal_IOU(pre_bboxes, bboxes):
    hw = pre_bboxes[:, 2:4] - pre_bboxes[:, :2]
    areas1 = hw.prod(dim=-1)
    hw = bboxes[:, 2:4] - bboxes[:, :2]
    areas2 = hw.prod(dim=-1)

    yx1 = torch.max(pre_bboxes[:, None, :2], bboxes[:, :2])
    yx2 = torch.min(pre_bboxes[:, None, 2:4], bboxes[:, 2:4])

    hw = yx2 - yx1
    hw = torch.max(hw, cuda(torch.Tensor([0])))
    areas_i = hw.prod(dim=-1)
    iou = areas_i / (areas1[:, None] + areas2 - areas_i)
    return iou


def bbox2loc(anchor, bbox):
    c_hw = anchor[..., 2:4] - anchor[..., 0:2]
    c_yx = anchor[..., :2] + c_hw / 2
    hw = bbox[..., 2:4] - bbox[..., 0:2]
    yx = bbox[..., :2] + hw / 2
    t_yx = (yx - c_yx) / c_hw
    t_hw = torch.log(hw / c_hw)
    return torch.cat([t_yx, t_hw], dim=1)


class AnchorTargetCreator(object):
    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        if bbox.shape[0] == 0:
            inds = torch.randperm(anchor.shape[0])[:self.n_sample]
            label = cuda(torch.zeros(inds.shape[0], dtype=torch.int64))
            indsP = cuda(torch.tensor([], dtype=torch.int64))
            loc = cuda(torch.zeros((0, 4), dtype=torch.float32))
            return inds, label, indsP, loc

        IOU = cal_IOU(anchor, bbox)

        iou, inds_box = IOU.max(dim=1)

        indsP1 = iou >= self.pos_iou_thresh
        indsN = iou < self.neg_iou_thresh

        t, _ = IOU.max(dim=0)
        t = IOU == t
        indsP2 = t.sum(dim=1) > 0
        if True:
            inds_gt_box = t.argmax(dim=1)
            inds_box[indsP2] = inds_gt_box[indsP2]
            # print('***************  ATC true')
        indsP = indsP1 | indsP2

        if False:
            indsN = indsN & (~indsP2)  # 小于neg_iou_thresh,但有最大匹配，为正样本
        else:
            indsP = indsP & (~indsN)  # 注意，这里是个参数，小于neg_iou_thresh,为负样本

        t = torch.arange(indsP.shape[0])
        indsP = t[indsP]
        indsN = t[indsN]
        p_num = indsP.shape[0]
        n_num = indsN.shape[0]
        n_pos = int(min((self.n_sample * self.pos_ratio), p_num))
        n_neg = int(min(self.n_sample - n_pos, n_num))
        indsP = indsP[torch.randperm(p_num)[:n_pos]]
        indsN = indsN[torch.randperm(n_num)[:n_neg]]

        anchor = anchor[indsP]
        bbox = bbox[inds_box[indsP]]

        loc = bbox2loc(anchor, bbox)

        label = cuda(torch.zeros(n_pos + n_neg, dtype=torch.int64))
        label[:n_pos] = 1
        inds = torch.cat([indsP, indsN], dim=0)

        return inds, label, indsP, loc
