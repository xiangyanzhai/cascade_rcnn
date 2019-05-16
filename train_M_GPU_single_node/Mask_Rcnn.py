# !/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
# torch.backends.cudnn.benchmark =True
from cascade_rcnn.tool.config import Config

from cascade_rcnn.tool.get_anchors import get_anchors
from cascade_rcnn.tool.torch_ATC_FPN import AnchorTargetCreator
from cascade_rcnn.tool.torch_PC_FPN import ProposalCreator
from cascade_rcnn.tool.torch_PTC_mask import ProposalTargetCreator
from maskrcnn_benchmark.layers import ROIAlign
from cascade_rcnn.tool.resnet import resnet101
from cascade_rcnn.tool.FPN_net import FPN_net
from cascade_rcnn.tool.FPN_RPN import RPN_net
from cascade_rcnn.tool.FPN_Fast import Fast_net
from cascade_rcnn.tool.Mask_net import Mask_net
import torch.nn.functional as F
from cascade_rcnn.tool.read_Data_mask import Read_Data
from torch.utils.data import Dataset, DataLoader

ce_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCEWithLogitsLoss()
roialign_list_7 = [ROIAlign((7, 7), 1 / 4., 2), ROIAlign((7, 7), 1 / 8., 2), ROIAlign((7, 7), 1 / 16., 2),
                   ROIAlign((7, 7), 1 / 32., 2)]
roialign_list_14 = [ROIAlign((14, 14), 1 / 4., 2), ROIAlign((14, 14), 1 / 8., 2), ROIAlign((14, 14), 1 / 16., 2),
                    ROIAlign((14, 14), 1 / 32., 2)]

roialign_28 = ROIAlign((28, 28), 1 / 1., 2)


def SmoothL1Loss(net_loc_train, loc, sigma, num):
    t = torch.abs(net_loc_train - loc)
    a = t[t < 1]
    b = t[t >= 1]
    loss1 = (a * sigma) ** 2 / 2
    loss2 = b - 0.5 / sigma ** 2
    loss = (loss1.sum() + loss2.sum()) / num
    return loss


class Mask_Rcnn(nn.Module):
    def __init__(self, config):
        super(Mask_Rcnn, self).__init__()
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
        self.mask_net = Mask_net(256, config.num_cls)
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

    def fast_train_data(self, loc, score, anchor, img_size, bboxes, map_HW, masks):
        roi = self.PC(loc, score, anchor, img_size, map_HW)
        roi, loc, label, inter, target_inds = self.PTC(roi, bboxes[:, :4], bboxes[:, -1].long())
        area = roi[:, 2:] - roi[:, :2] + 1
        area = area.prod(dim=-1)
        roi_inds = torch.floor(4.0 + torch.log(area ** 0.5 / 224.0) / np.log(2.0))
        roi_inds = roi_inds.clamp(2, 5) - 2
        roi = torch.cat([torch.zeros(roi.shape[0], 1).cuda(), roi], dim=-1)

        area = inter[:, 2:] - inter[:, :2] + 1
        area = area.prod(dim=-1)
        inter_inds = torch.floor(4.0 + torch.log(area ** 0.5 / 224.0) / np.log(2.0))
        inter_inds = inter_inds.clamp(2, 5) - 2

        target_inds = target_inds.contiguous().view(-1, 1)
        target_norm = torch.cat([target_inds.float(), inter], dim=-1)

        inter = torch.cat([torch.zeros(inter.shape[0], 1).cuda(), inter], dim=-1)

        H, W = img_size

        mask_H, mask_W = masks.shape[1:]
        masks = masks[..., None]
        masks = masks.permute(0, 3, 1, 2)

        target_norm[:, 1:] = target_norm[:, 1:] / torch.tensor([W, H, W, H], dtype=torch.float32).cuda() * torch.tensor(
            [mask_W, mask_H, mask_W, mask_H], dtype=torch.float32).cuda()

        target = roialign_28(masks, target_norm)

        return roi, roi_inds, loc, label, inter, inter_inds, target[:, 0]

    def fast_loss(self, fast_logits, fast_loc, label, loc, mask, target):
        fast_num = label.shape[0]
        fast_num_P = loc.shape[0]
        a = torch.arange(fast_num_P)
        b = label[:fast_num_P].long()
        fast_loc_train = fast_loc[a, b]
        fast_cls_loss = ce_loss(fast_logits, label.long())
        fast_box_loss = SmoothL1Loss(fast_loc_train, loc, 1.0, float(fast_num))
        # print(target.max(),target.min())
        mask_loss = bce_loss(mask[a, b], target)
        self.c = fast_cls_loss
        self.d = fast_box_loss
        self.fast_num = fast_num
        self.fast_num_P = fast_num_P
        self.f = mask_loss
        return fast_cls_loss, fast_box_loss, mask_loss

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

    def pooling(self, P, roi, roi_inds, size):
        x = []
        inds = []
        index = torch.arange(roi.shape[0]).cuda()

        for i in range(4):
            t = roi_inds == i
            if size == 7:
                x.append(roialign_list_7[i](P[i], roi[t]))
            else:

                x.append(roialign_list_14[i](P[i], roi[t]))
            inds.append(index[t])

        x = torch.cat(x, dim=0)
        inds = torch.cat(inds, dim=0)
        inds = inds.argsort()
        x = x[inds]
        return x

    def get_loss(self, x, bboxes, num_b, H, W, masks):

        x = x.view(-1)[:H * W * 3].view(H, W, 3)
        bboxes = bboxes[:num_b]
        masks = masks.view(-1)[:num_b * H * W].view(num_b, H, W)

        inds = bboxes[:, -1] >= 0
        bboxes = bboxes[inds]
        masks = masks[inds]

        x = x.float().cuda()
        bboxes = bboxes.cuda()
        masks = masks.float().cuda()
        x, bboxes = self.process_im(x, bboxes)
        x = x - self.Mean.cuda()
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
        tanchors = torch.cat(tanchors, dim=0).cuda()

        rpn_cls_loss, rpn_box_loss = self.rpn_loss(rpn_logits, rpn_loc, bboxes, tanchors, img_size)
        roi, roi_inds, loc, label, inter, inter_inds, target = self.fast_train_data(rpn_loc.data,
                                                                                    F.softmax(rpn_logits.data, dim=-1)[
                                                                                    :, 1],
                                                                                    tanchors,
                                                                                    img_size, bboxes, map_HW, masks)
        x = self.pooling(P, roi, roi_inds, 7)
        fast_logits, fast_loc = self.fast(x)
        fast_cls_loss, fast_box_loss = self.fast_loss_box(fast_logits, fast_loc, label, loc)
        fast_num_P = loc.shape[0]
        if fast_num_P > 0:
            x_mask = self.pooling(P, inter, inter_inds, 14)
            mask = self.mask_net(x_mask)
            a = torch.arange(fast_num_P)
            b = label[:fast_num_P].long()
            mask_loss = bce_loss(mask[a, b], target)
        else:
            with torch.no_grad():
                x_mask = self.pooling(P, roi[:1], roi_inds[:1], 14)
            mask = self.mask_net(x_mask)
            mask_loss = (mask[0][0] - mask[0][0].data) ** 2 / 2
            mask_loss = mask_loss.mean()
        self.f = mask_loss
        return rpn_cls_loss + rpn_box_loss + fast_cls_loss + fast_box_loss + mask_loss

    def forward(self, imgs, bboxes, num_b, num_H, num_W, masks):
        loss = list(map(self.get_loss, imgs, bboxes, num_b, num_H, num_W, masks))
        return sum(loss)

    def fast_loss_box(self, fast_logits, fast_loc, label, loc):
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
    masks = []
    for i in range(m):
        imgs.append(batch[i][0].resize_(max_H, max_W, 3)[None])
        bboxes.append(batch[i][1].resize_(max_b, 5)[None])
        masks.append(batch[i][-1].resize_(max_b, max_H, max_W)[None])

    imgs = torch.cat(imgs, dim=0)
    bboxes = torch.cat(bboxes, dim=0)
    masks = torch.cat(masks, dim=0)

    return imgs, bboxes, torch.tensor(num_b, dtype=torch.int64), torch.tensor(num_H, dtype=torch.int64), torch.tensor(
        num_W, dtype=torch.int64), masks


from datetime import datetime


def average_gradients(model, size):
    """ Gradient averaging. """

    for name, param in list(model.named_parameters()):
        # print(type(param.grad.data))
        # print(name,param.requires_grad,param.grad is None)
        dist.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM, )
        param.grad.data /= size


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
    model.eval()
    model_dic = model.state_dict()

    pretrained_dict = torch.load(pre_model_file, map_location='cpu')
    pretrained_dict = {'features.' + k: v for k, v in pretrained_dict.items() if 'features.' + k in model_dic}

    print('pretrained_dict  *******', len(pretrained_dict))
    model_dic.update(pretrained_dict)
    model.load_state_dict(model_dic)

    if step > 0:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        print(model_file)
    else:
        print(pre_model_file)

    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(), device_ids=[local_rank], output_device=local_rank,
        # this should be removed if we update BatchNorm stats
        broadcast_buffers=False,
    )

    lr = config.lr * config.batch_size_per_GPU
    if step >= 60000 * x:
        lr = lr / 10
    if step >= 80000 * x:
        lr = lr / 10
    print('lr               ******************', lr)
    print('weight_decay     ******************', config.weight_decay)

    train_params = list(model.parameters())
    bias_p = []
    weight_p = []
    bn_weight_p = []
    print(len(train_params))
    for name, p in model.named_parameters():
        print(name, p.shape)
        # if 'bias' in name:
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
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=config.batch_size_per_GPU, sampler=train_sampler,
                            collate_fn=func, drop_last=True, pin_memory=True, num_workers=16)
    epochs = 10000
    flag = False
    print('start:  step=', step)
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        for imgs, bboxes, num_b, num_H, num_W, masks in dataloader:

            loss = model(imgs, bboxes, num_b, num_H, num_W, masks)
            loss = loss / imgs.shape[0]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 5, norm_type=2)
            opt.step()

            if step % 20 == 0 and local_rank == 0:
                print(datetime.now(), 'loss:%.4f' % (loss.data), step)
                pass
            step += 1

            if (step == int(60000 * x) or step == int(80000 * x)):
                for param_group in opt.param_groups:
                    param_group['lr'] = param_group['lr'] / 10
                    print('***************************', param_group['lr'], local_rank)

            if ((step <= 10000 and step % 1000 == 0) or step % 5000 == 0 or step == 1) and local_rank == 0:
                torch.save(model.module.state_dict(),
                           './models/Mask_Rcnn_%dx_%d_1_%d.pth' % (x, step, local_rank))
            if step >= 90010 * x:
                flag = True
                break
        if flag:
            break
    if local_rank == 0:
        torch.save(model.module.state_dict(),
                   './models/Mask_Rcnn_%dx_final_1_%d.pth' % (x, local_rank))


if __name__ == "__main__":
    Mean = [123.68, 116.78, 103.94]
    path = '/home/zhai/PycharmProjects/Demo35/cascade_rcnn/data_preprocess/'
    Bboxes = [path + 'coco_bboxes_2017.pkl']
    img_paths = [path + 'coco_imgpaths_2017.pkl']
    masks = [path + 'coco_mask_2017.pkl']
    files = [img_paths, Bboxes, masks]

    config = Config(True, Mean, files, num_cls=80, lr=0.00125, weight_decay=0.0001, batch_size_per_GPU=2, gpus=2,
                    img_max=1333, img_min=800,
                    anchor_scales=[[32], [64], [128], [256], [512]],
                    anchor_ratios=[[0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2]], fast_n_sample=512,
                    bias_lr_factor=2,
                    roi_min_size=[4, 8, 16, 32, 64],
                    roi_train_pre_nms=12000,
                    roi_train_post_nms=2000,
                    roi_test_pre_nms=6000,
                    roi_test_post_nms=1000)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    assert num_gpus == config.gpus
    print('--------GPUs--------', num_gpus)
    step = 0
    model = Mask_Rcnn
    x = 4
    pre_model_file = '/home/zhai/PycharmProjects/Demo35/pytorch_Faster_tool/resnet_caffe/resnet101-caffe.pth'
    model_file = ''
    train_dist(model, config, step, x, pre_model_file, model_file=model_file)
