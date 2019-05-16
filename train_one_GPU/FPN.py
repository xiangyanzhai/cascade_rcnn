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


import cv2
# torch.backends.cudnn.benchmark =True
from cascade_rcnn.tool.config import Config

from cascade_rcnn.tool.get_anchors import get_anchors
from cascade_rcnn.tool.torch_ATC_FPN import AnchorTargetCreator
from cascade_rcnn.tool.torch_PC_FPN import ProposalCreator
from cascade_rcnn.tool.torch_PTC_test import ProposalTargetCreator
from maskrcnn_benchmark.layers import ROIAlign
from cascade_rcnn.tool.resnet import resnet101
from cascade_rcnn.tool.FPN_net import FPN_net
from cascade_rcnn.tool.FPN_RPN import RPN_net
from cascade_rcnn.tool.FPN_Fast import Fast_net
import torch.nn.functional as F
from cascade_rcnn.tool.read_Data import Read_Data
from torch.utils.data import DataLoader


def draw_gt(im, gt):
    im = im.astype(np.uint8).copy()
    boxes = gt.astype(np.int32)
    print(im.max(), im.min(), im.shape)
    for box in boxes:
        # print(box)
        x1, y1, x2, y2 = box[:4]

        im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255))
        print(box[-1])
    im = im.astype(np.uint8)
    cv2.imshow('a', im)
    cv2.waitKey(2000)
    return im


ce_loss = nn.CrossEntropyLoss()

roialign_list = [ROIAlign((7, 7), 1 / 4., 2), ROIAlign((7, 7), 1 / 8., 2), ROIAlign((7, 7), 1 / 16., 2),
                 ROIAlign((7, 7), 1 / 32., 2)]


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

    def rpn_loss(self, rpn_logits, rpn_loc, bboxes, tanchors, img_size):
        inds, label, indsP, loc = self.ATC(bboxes, tanchors, img_size)

        rpn_logits_train = rpn_logits[inds]
        rpn_loc_train = rpn_loc[indsP]

        rpn_cls_loss = ce_loss(rpn_logits_train, label)
        rpn_box_loss = SmoothL1Loss(rpn_loc_train, loc, 3.0, float(label.shape[0]))
        self.a = rpn_cls_loss
        self.b = rpn_box_loss
        return rpn_cls_loss, rpn_box_loss

    def fast_train_data(self, loc, score, anchor, img_size, bboxes, map_HW):
        roi = self.PC(loc, score, anchor, img_size, map_HW)
        roi, loc, label = self.PTC(roi, bboxes[:, :4], bboxes[:, -1].long())
        area = roi[:, 2:] - roi[:, :2] + 1
        area = area.prod(dim=-1)
        roi_inds = torch.floor(4.0 + torch.log(area ** 0.5 / 224.0) / np.log(2.0))
        roi_inds = roi_inds.clamp(2, 5) - 2

        roi = torch.cat([cuda(torch.zeros(roi.shape[0], 1)), roi], dim=-1)
        return roi, roi_inds, loc, label

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

    def get_loss(self, x, bboxes, num_b, H, W):
        x = x.view(-1)[:H * W * 3].view(H, W, 3)
        bboxes = bboxes[:num_b]
        inds = bboxes[:, -1] >= 0
        bboxes = bboxes[inds]
        x = cuda(x.float())
        bboxes = cuda(bboxes)
        x, bboxes = self.process_im(x, bboxes)
        # draw_gt(x[0].cpu().numpy(),bboxes.cpu().numpy())
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

        rpn_cls_loss, rpn_box_loss = self.rpn_loss(rpn_logits, rpn_loc, bboxes, tanchors, img_size)
        roi, roi_inds, loc, label = self.fast_train_data(rpn_loc.data, F.softmax(rpn_logits.data, dim=-1)[:, 1],
                                                         tanchors,
                                                         img_size, bboxes, map_HW)

        x = self.pooling(P, roi, roi_inds)
        fast_logits, fast_loc = self.fast(x)
        fast_cls_loss, fast_box_loss = self.fast_loss(fast_logits, fast_loc, label, loc)
        return rpn_cls_loss + rpn_box_loss + fast_cls_loss + fast_box_loss

    def forward(self, imgs, bboxes, num_b, num_H, num_W):
        loss = list(map(self.get_loss, imgs, bboxes, num_b, num_H, num_W))
        return sum(loss)


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
    dataset = Read_Data(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size_per_GPU, collate_fn=func,
                            shuffle=True, drop_last=True, pin_memory=True)
    model = model(config)
    print(model)
    model.eval()
    model_dic = model.state_dict()
    pretrained_dict = torch.load(pre_model_file, map_location='cpu')
    pretrained_dict = {'features.' + k: v for k, v in pretrained_dict.items() if 'features.' + k in model_dic}
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
    bias_p = []
    weight_p = []

    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p.append(p)
        else:
            weight_p.append(p)
    print(len(weight_p), len(bias_p))
    lr = config.lr * config.batch_size_per_GPU
    if lr >= 60000 * x:
        lr = lr / 10
    if lr >= 80000 * x:
        lr = lr / 10
    print('lr        ******************', lr)
    opt = torch.optim.SGD(
        [{'params': weight_p, 'weight_decay': config.weight_decay, 'lr': lr},
         {'params': bias_p, 'lr': lr * config.bias_lr_factor}],
        momentum=0.9, )

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
                    print('*********************************', param_group['lr'])

            if (step <= 10000 and step % 1000 == 0) or step % 5000 == 0 or step == 1:
                torch.save(model.state_dict(), './models/FPN_101_%d_1.pth' % step)

            if step >= 90010 * x:
                flag = True
                break
        if flag:
            break
    torch.save(model.state_dict(), './models/FPN_101_final_1.pth' % step)


if __name__ == "__main__":
    Mean = [123.68, 116.78, 103.94]
    path = '/home/zhai/PycharmProjects/Demo35/cascade_rcnn/data_preprocess/'
    Bboxes = [path + 'Bboxes_07.pkl', path + 'Bboxes_12.pkl']
    img_paths = [path + 'img_paths_07.pkl', path + 'img_paths_12.pkl']
    files = [img_paths, Bboxes]

    config = Config(True, Mean, files, num_cls=20, lr=0.00125, weight_decay=0.0001, batch_size_per_GPU=1,
                    img_max=1333, img_min=800,
                    anchor_scales=[[32], [64], [128], [256], [512]],
                    anchor_ratios=[[0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2]], fast_n_sample=512,
                    bias_lr_factor=2,
                    roi_min_size=[4, 8, 16, 32, 64],
                    roi_train_pre_nms=12000,
                    roi_train_post_nms=2000,
                    roi_test_pre_nms=6000,
                    roi_test_post_nms=1000)

    step = 0
    model = Faster_Rcnn
    x = 1
    pre_model_file = '/home/zhai/PycharmProjects/Demo35/pytorch_Faster_tool/resnet_caffe/resnet101-caffe.pth'
    model_file = None
    train(model, config, step, x, pre_model_file, model_file=model_file)
