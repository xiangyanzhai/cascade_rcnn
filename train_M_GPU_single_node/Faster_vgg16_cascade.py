# !/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

is_gpu = torch.cuda.is_available()


def cuda(x):
    if is_gpu:
        x = x.cuda()
    return x


# torch.backends.cudnn.benchmark =True
# torch.backends.cudnn.deterministic = True
from cascade_rcnn.tool.config import Config

from cascade_rcnn.tool.get_anchors import get_anchors
from cascade_rcnn.tool.torch_ATC_test import AnchorTargetCreator
from cascade_rcnn.tool.torch_PC_test import ProposalCreator
from cascade_rcnn.tool.torch_PTC_test import ProposalTargetCreator
from maskrcnn_benchmark.layers import ROIAlign
from torchvision.models import vgg16
from cascade_rcnn.tool.RPN_net import RPN_net
from cascade_rcnn.tool.Fast_net import Fast_net
import torch.nn.functional as F
from cascade_rcnn.tool.read_Data import Read_Data
from torch.utils.data import DataLoader

ce_loss = nn.CrossEntropyLoss()
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
        self.ATC = AnchorTargetCreator(n_sample=config.rpn_n_sample, pos_iou_thresh=config.rpn_pos_iou_thresh,
                                       neg_iou_thresh=config.rpn_neg_iou_thresh, pos_ratio=config.rpn_pos_ratio)
        self.PC = ProposalCreator(nms_thresh=config.roi_nms_thresh,
                                  n_train_pre_nms=config.roi_train_pre_nms,
                                  n_train_post_nms=config.roi_train_post_nms,
                                  n_test_pre_nms=config.roi_test_pre_nms, n_test_post_nms=config.roi_test_post_nms,
                                  min_size=config.roi_min_size)
        self.PTC_1 = ProposalTargetCreator(n_sample=config.fast_n_sample,
                                           pos_ratio=config.fast_pos_ratio, pos_iou_thresh=config.fast_pos_iou_thresh,
                                           neg_iou_thresh_hi=config.fast_neg_iou_thresh_hi,
                                           neg_iou_thresh_lo=config.fast_neg_iou_thresh_lo)
        self.PTC_2 = ProposalTargetCreator(n_sample=config.fast_n_sample,
                                           pos_ratio=config.fast_pos_ratio, pos_iou_thresh=0.6,
                                           neg_iou_thresh_hi=0.6,
                                           neg_iou_thresh_lo=config.fast_neg_iou_thresh_lo)

        self.PTC_3 = ProposalTargetCreator(n_sample=config.fast_n_sample,
                                           pos_ratio=config.fast_pos_ratio, pos_iou_thresh=0.7,
                                           neg_iou_thresh_hi=0.7,
                                           neg_iou_thresh_lo=config.fast_neg_iou_thresh_lo)

        self.features = vgg16().features[:-1]
        self.rpn = RPN_net(512, self.num_anchor)

        self.fast_1 = Fast_net(config.num_cls, 512 * 7 * 7, 2048)
        self.fast_2 = Fast_net(config.num_cls, 512 * 7 * 7, 2048)
        self.fast_3 = Fast_net(config.num_cls, 512 * 7 * 7, 2048)
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.fast_num = 0
        self.fast_num_P = 0

        self.loc_std1 = [1. / 10, 1. / 10, 1. / 5, 1. / 5]
        self.loc_std2 = [1. / 20, 1. / 20, 1. / 10, 1. / 10]
        self.loc_std3 = [1. / 30, 1. / 30, 1. / 15, 1. / 15]
        self.loss_weights = [1.0, 0.5, 0.25]

    def rpn_loss(self, rpn_logits, rpn_loc, bboxes, tanchors, img_size):
        inds, label, indsP, loc = self.ATC(bboxes, tanchors, img_size)

        rpn_logits_train = rpn_logits[inds]
        rpn_loc_train = rpn_loc[indsP]
        rpn_cls_loss = ce_loss(rpn_logits_train, label)
        rpn_box_loss = SmoothL1Loss(rpn_loc_train, loc, 3.0, 240.0)
        self.a = rpn_cls_loss
        self.b = rpn_box_loss
        return rpn_cls_loss, rpn_box_loss

    def fast_train_data(self, roi, bboxes, PTC, loc_std):
        roi, loc, label = PTC(roi, bboxes[:, :4], bboxes[:, -1].long(), loc_normalize_std=loc_std)
        roi_inds = cuda(torch.zeros((roi.size()[0], 1)))
        roi = torch.cat([roi_inds, roi], dim=1)
        return roi, loc, label

    def fast_loss(self, fast_logits, fast_loc, label, loc):
        fast_num = label.shape[0]
        fast_num_P = loc.shape[0]
        fast_loc_train = fast_loc[torch.arange(fast_num_P), label[:fast_num_P].long()]

        fast_cls_loss = ce_loss(fast_logits, label.long())
        fast_box_loss = SmoothL1Loss(fast_loc_train, loc, 1.0, float(fast_num))

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
        x = x.permute(0, 3, 1, 2)
        img_size = x.shape[2:]

        x = self.features(x)
        rpn_logits, rpn_loc = self.rpn(x)
        map_H, map_W = x.shape[2:]
        tanchors = self.anchors[:map_H, :map_W]
        tanchors = cuda(tanchors.contiguous().view(-1, 4))

        rpn_cls_loss, rpn_box_loss = self.rpn_loss(rpn_logits, rpn_loc, bboxes, tanchors, img_size)

        cls_loss = 0
        box_loss = 0

        with torch.no_grad():
            roi = self.PC(rpn_loc.data, F.softmax(rpn_logits.data, dim=-1)[:, 1], tanchors, img_size)
            roi, loc, label = self.fast_train_data(roi, bboxes, self.PTC_1, self.loc_std1)

        xx = roialign(x, roi)
        fast_logits, fast_loc = self.fast_1(xx)
        fast_cls_loss, fast_box_loss = self.fast_loss(fast_logits, fast_loc, label, loc)
        cls_loss += fast_cls_loss * self.loss_weights[0]
        box_loss += fast_box_loss * self.loss_weights[0]

        with torch.no_grad():
            fast_loc = fast_loc[:, 1:] * cuda(torch.tensor(self.loc_std1))
            score = F.softmax(fast_logits, dim=-1)[:, 1:]
            _, inds = score.max(dim=-1)
            t = torch.arange(score.shape[0])
            fast_loc = fast_loc[t, inds]
            roi = self.loc2bbox(fast_loc, roi[:, 1:])
            roi = self.filter_bboxes(roi, img_size, self.config.roi_min_size)
            roi, loc, label = self.fast_train_data(roi, bboxes, self.PTC_2, self.loc_std2)

        xx = roialign(x, roi)
        fast_logits, fast_loc = self.fast_2(xx)
        fast_cls_loss, fast_box_loss = self.fast_loss(fast_logits, fast_loc, label, loc)
        cls_loss += fast_cls_loss * self.loss_weights[1]
        box_loss += fast_box_loss * self.loss_weights[1]

        with torch.no_grad():
            fast_loc = fast_loc[:, 1:] * cuda(torch.tensor(self.loc_std2))
            score = F.softmax(fast_logits, dim=-1)[:, 1:]
            _, inds = score.max(dim=-1)
            t = torch.arange(score.shape[0])
            fast_loc = fast_loc[t, inds]
            roi = self.loc2bbox(fast_loc, roi[:, 1:])
            roi = self.filter_bboxes(roi, img_size, self.config.roi_min_size)
            roi, loc, label = self.fast_train_data(roi, bboxes, self.PTC_3, self.loc_std3)

        xx = roialign(x, roi)
        fast_logits, fast_loc = self.fast_3(xx)
        fast_cls_loss, fast_box_loss = self.fast_loss(fast_logits, fast_loc, label, loc)
        cls_loss += fast_cls_loss * self.loss_weights[2]
        box_loss += fast_box_loss * self.loss_weights[2]

        self.c = cls_loss
        self.d = box_loss
        self.fast_num = roi.shape[0]
        self.fast_num_P = loc.shape[0]
        return rpn_cls_loss + rpn_box_loss + cls_loss + box_loss

    def forward(self, imgs, bboxes, num_b, num_H, num_W):
        loss = list(map(self.get_loss, imgs, bboxes, num_b, num_H, num_W))
        return sum(loss)

    def filter_bboxes(self, roi, img_size, roi_min_size):
        h, w = img_size
        roi[:, slice(0, 4, 2)] = torch.clamp(roi[:, slice(0, 4, 2)], 0, w)
        roi[:, slice(1, 4, 2)] = torch.clamp(roi[:, slice(1, 4, 2)], 0, h)
        hw = roi[:, 2:4] - roi[:, :2]
        inds = hw >= roi_min_size
        inds = inds.all(dim=-1)
        roi = roi[inds]
        return roi

    def loc2bbox(self, pre_loc, anchor):
        c_hw = anchor[..., 2:4] - anchor[..., 0:2]
        c_yx = anchor[..., :2] + c_hw / 2
        yx = pre_loc[..., :2] * c_hw + c_yx
        hw = torch.exp(pre_loc[..., 2:4]) * c_hw
        yx1 = yx - hw / 2
        yx2 = yx + hw / 2
        bboxes = torch.cat((yx1, yx2), dim=-1)
        return bboxes



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


def train_dist(model, config, step, x, pre_model_file, model_file=None):
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    local_rank = args.local_rank
    print('******************* local_rank', local_rank)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    assert torch.distributed.is_initialized()
    batch_size = config.gpus * config.batch_size_per_GPU
    print('--------batch_size--------', batch_size)

    model = model(config)
    print(model)
    model.eval()
    model_dic = model.state_dict()

    pretrained_dict = torch.load(pre_model_file, map_location='cpu')
    a = pretrained_dict['classifier.0.weight']
    b = pretrained_dict['classifier.0.bias']
    c = pretrained_dict['classifier.3.weight']
    d = pretrained_dict['classifier.3.bias']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dic}
    print(len(pretrained_dict))
    model_dic.update(pretrained_dict)
    print(list(model_dic.keys()))
    # model_dic['fast.fast_head.0.weight'] = a
    # model_dic['fast.fast_head.0.bias'] = b
    # model_dic['fast.fast_head.2.weight'] = c
    # model_dic['fast.fast_head.2.bias'] = d
    model.load_state_dict(model_dic)

    if step > 0:

        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        print(model_file)
    else:
        print(pre_model_file)

    parameters = list(model.parameters())
    for i in range(8):
        parameters[i].requires_grad = False


    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(), device_ids=[local_rank], output_device=local_rank,
        # this should be removed if we update BatchNorm stats
        broadcast_buffers=False,
    )

    train_params = list(model.parameters())[8:]

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

    opt = torch.optim.SGD([{'params': weight_p, 'weight_decay': config.weight_decay, 'lr': lr},
                           {'params': bias_p, 'lr': lr * config.bias_lr_factor}],
                          momentum=0.9, )

    epochs = 10000
    flag = False
    dataset = Read_Data(config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=config.batch_size_per_GPU, sampler=train_sampler,
                            collate_fn=func, drop_last=True, pin_memory=True)
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)

        for imgs, bboxes, num_b, num_H, num_W in dataloader:

            loss = model(imgs, bboxes, num_b, num_H, num_W)
            loss = loss / imgs.shape[0]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 35, norm_type=2)
            opt.step()

            # torch.cuda.empty_cache()
            if step % 20 == 0 and local_rank == 0:
                print(datetime.now(), 'loss:%.4f' % (loss), step)
            step += 1

            if (step == int(60000 * x) or step == int(80000 * x)):
                for param_group in opt.param_groups:
                    param_group['lr'] = param_group['lr'] / 10
                    print('***************************', param_group['lr'], local_rank)
            if ((step <= 10000 and step % 1000 == 0) or step % 5000 == 0 or step == 1) and local_rank == 0:
                torch.save(model.module.state_dict(), './models/vgg16_cascade_%dx_%d_1_%d.pth' % (x, step, local_rank))
            if step >= 90010 * x:
                flag = True
                break
        if flag:
            break
    if local_rank == 0:
        torch.save(model.module.state_dict(), './models/vgg16_cascade_%dx_final_1_%d.pth' % (x,  local_rank))


if __name__ == "__main__":
    Mean = [123.68, 116.78, 103.94]
    path = '/home/zhai/PycharmProjects/Demo35/cascade_rcnn/data_preprocess/'
    Bboxes = [path + 'Bboxes_07.pkl', path + 'Bboxes_12.pkl']
    img_paths = [path + 'img_paths_07.pkl', path + 'img_paths_12.pkl']


    files = [img_paths, Bboxes]
    config = Config(True, Mean, files, lr=0.001, weight_decay=0.0005,
                    gpus=2, batch_size_per_GPU=1,
                    img_max=1000, img_min=600,roi_min_size=16,
                    bias_lr_factor=2)

    step = 0
    model = Faster_Rcnn
    x = 1
    pre_model_file = '/home/zhai/PycharmProjects/Demo35/py_Faster_tool/pre_model/vgg16_cf.pth'
    model_file = ''
    train_dist(model, config, step, x, pre_model_file, model_file=model_file)
