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


#     return iou

def bbox2loc(anchor, bbox):
    c_hw = anchor[..., 2:4] - anchor[..., 0:2]
    c_yx = anchor[..., :2] + c_hw / 2
    hw = bbox[..., 2:4] - bbox[..., 0:2]
    yx = bbox[..., :2] + hw / 2
    t_yx = (yx - c_yx) / c_hw
    t_hw = torch.log(hw / c_hw)
    return torch.cat([t_yx, t_hw], dim=1)


class ProposalTargetCreator(object):
    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        if bbox.shape[0] == 0:
            inds = torch.randperm(roi.shape[0])[:self.n_sample]
            roi = roi[inds]
            loc = cuda(torch.zeros((0, 4), dtype=torch.float32))
            label = cuda(torch.zeros(roi.shape[0], dtype=torch.int64))
            return roi, loc, label

        roi = torch.cat([roi, bbox], dim=0)
        IOU = cal_IOU(roi, bbox)
        iou, inds_box = IOU.max(dim=1)

        indsP = iou >= self.pos_iou_thresh
        indsN = (iou >= self.neg_iou_thresh_lo) & (iou < self.neg_iou_thresh_hi)

        t = torch.arange(indsP.shape[0])
        indsP = t[indsP]
        indsN = t[indsN]
        p_num = indsP.shape[0]
        n_num = indsN.shape[0]
        n_pos = int(min((self.n_sample * self.pos_ratio), p_num))
        n_neg = int(min(self.n_sample - n_pos, n_num))
        indsP = indsP[torch.randperm(p_num)[:n_pos]]
        indsN = indsN[torch.randperm(n_num)[:n_neg]]

        roiP = roi[indsP]
        roiN = roi[indsN]

        inds_box = inds_box[indsP]
        loc = bbox2loc(roiP, bbox[inds_box])
        loc = (loc - cuda(torch.tensor(loc_normalize_mean))) / cuda(torch.tensor(loc_normalize_std))
        label = label[inds_box] + 1

        roi = torch.cat([roiP, roiN], dim=0)

        label = torch.cat([label, cuda(torch.zeros(n_neg, dtype=torch.int64))], dim=0)
        return roi, loc, label
