# !/usr/bin/python
# -*- coding:utf-8 -*-
import torch
from maskrcnn_benchmark.layers import nms as _box_nms
is_gpu = torch.cuda.is_available()


def cuda(x):
    if is_gpu:
        x = x.cuda()
    return x
# from .bounding_box import BoxList
#
# from maskrcnn_benchmark.layers import nms as _box_nms

iou_thresh = None


def loc2bbox(pre_loc, anchor):
    c_hw = anchor[..., 2:4] - anchor[..., 0:2]
    c_yx = anchor[..., :2] + c_hw / 2
    yx = pre_loc[..., :2] * c_hw + c_yx
    hw = torch.exp(pre_loc[..., 2:4]) * c_hw
    yx1 = yx - hw / 2
    yx2 = yx + hw / 2
    bboxes = torch.cat((yx1, yx2), dim=-1)
    return bboxes


def map_func(bbox, score, cls):
    score, inds = score.sort(descending=True)
    bbox = bbox[inds]
    keep = _box_nms(bbox, score, iou_thresh)
    bbox = bbox[keep]
    score = score[keep]
    cls = cuda(torch.zeros((bbox.shape[0], 1))) + cls
    score = score.view(-1, 1)

    return torch.cat([bbox, score, cls], dim=1)


def predict(fast_loc, fast_score, roi, img_H, img_W, iou_thresh_=0.3, c_thresh=1e-3):
    # fast_loc m*cls*4
    # fast_score m*cls
    global iou_thresh
    iou_thresh = iou_thresh_
    fast_loc = fast_loc.permute(1, 0, 2)[1:]
    bboxes = loc2bbox(fast_loc, roi)

    bboxes[..., slice(0, 4, 2)] = torch.clamp(bboxes[..., slice(0, 4, 2)], 0, img_W)
    bboxes[..., slice(1, 4, 2)] = torch.clamp(bboxes[..., slice(1, 4, 2)], 0, img_H)
    fast_score = fast_score.t()[1:]

    pre = list(map(map_func, bboxes, fast_score, range(bboxes.shape[0])))

    pre = torch.cat(pre, dim=0)

    _, inds = pre[:, -2].sort(descending=True)
    pre = pre[inds]
    inds = pre[:, -2] > c_thresh
    pre = pre[inds]
    # pre = pre[:, [1, 0, 3, 2, 4, 5]]
    return pre


if __name__ == "__main__":
    pass
