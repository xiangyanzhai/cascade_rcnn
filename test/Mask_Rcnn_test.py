# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch.nn as nn
import torch

is_gpu = torch.cuda.is_available()


def cuda(x):
    if is_gpu:
        x = x.cuda()
    return x


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
from cascade_rcnn.tool.faster_predict import predict
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

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

    def pooling(self, P, roi, roi_inds, size):
        x = []
        inds = []
        index = cuda(torch.arange(roi.shape[0]))

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

    def forward(self, x):

        x = cuda(x.float())

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
        roi, roi_inds = self.roi_layer(rpn_loc.data, F.softmax(rpn_logits.data, dim=-1)[:, 1], tanchors, img_size,
                                       map_HW)

        x = self.pooling(P, roi, roi_inds, 7)

        fast_logits, fast_loc = self.fast(x)
        fast_loc = fast_loc * cuda(torch.tensor([0.1, 0.1, 0.2, 0.2]))
        pre = predict(fast_loc, F.softmax(fast_logits, dim=-1), roi[:, 1:], img_size[0], img_size[1], iou_thresh_=0.5,
                      c_thresh=0.05)[:100]
        if pre.shape[0] == 0:
            return pre, cuda(torch.zeros((0, 28, 28)))

        roi = pre[:100]
        inds_b = roi[:, -1].long() + 1
        area = roi[:, 2:4] - roi[:, :2] + 1
        area = area.prod(dim=-1)
        roi_inds = torch.floor(4.0 + torch.log(area ** 0.5 / 224.0) / np.log(2.0))
        roi_inds = roi_inds.clamp(2, 5) - 2

        roi = torch.cat([cuda(torch.zeros(roi.shape[0], 1)), roi[:, :4]], dim=1)

        net_mask = self.pooling(P, roi, roi_inds, 14)
        net_mask = self.mask_net(net_mask)
        net_mask = torch.sigmoid(net_mask)
        inds_a = torch.arange(roi.shape[0])

        mask = net_mask[inds_a, inds_b]
        pre[:, :4] = pre[:, :4] / scale
        return pre, mask


from datetime import datetime
from sklearn.externals import joblib


def loadNumpyAnnotations(data):
    """
    Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
    :param  data (numpy.ndarray)
    :return: annotations (python nested list)
    """
    print('Converting ndarray to lists...')
    assert (type(data) == np.ndarray)
    print(data.shape)
    assert (data.shape[1] == 7)
    N = data.shape[0]
    ann = []
    for i in range(N):
        if i % 1000000 == 0:
            print('{}/{}'.format(i, N))

        ann += [{
            'image_id': int(data[i, 0]),
            'bbox': [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
            'score': data[i, 5],
            'category_id': int(data[i, 6]),
        }]

    return ann


import cv2
from pycocotools import mask as maskUtils


def loadNumpyAnnotations_mask(data, mask):
    global oh, ow

    t = {
        'image_id': int(data[0]),
        'bbox': [data[1], data[2], data[3], data[4]],
        'score': data[5],
        'category_id': int(data[6]),
    }

    res_mask = np.zeros((oh, ow), dtype=np.uint8, order='F')
    bbox = t['bbox']
    bbox = np.round(bbox)
    bbox = bbox.astype(np.int32)
    x1 = bbox[0]
    y1 = bbox[1]
    w = bbox[2]
    h = bbox[3]
    x2 = x1 + w
    y2 = y1 + h

    x1 = np.clip(x1, 0, ow)
    x2 = np.clip(x2, 0, ow)

    y1 = np.clip(y1, 0, oh)
    y2 = np.clip(y2, 0, oh)
    w = x2 - x1
    h = y2 - y1

    img = cv2.resize(mask, (w, h))
    img = np.round(img)
    img = img.astype(np.uint8)

    res_mask[y1:y2, x1:x2] = img
    tt = maskUtils.encode(res_mask)
    tt['counts'] = tt['counts'].decode('utf-8')
    t["segmentation"] = tt

    return t


import codecs
import json
import eval_coco_box
import eval_coco_segm


def test(model, config, model_file):
    global oh, ow
    catId2cls, cls2catId, catId2name = joblib.load(
        r'/home/zhai/PycharmProjects/Demo35/cascade_rcnn/data_preprocess/(catId2cls,cls2catId,catId2name).pkl')
    model = model(config)
    model.eval()
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    cuda(model)
    test_dir = r'/home/zhai/PycharmProjects/Demo35/dataset/coco/val2017/'
    names = os.listdir(test_dir)
    names = [name.split('.')[0] for name in names]
    names = sorted(names)

    i = 0
    mm = 10
    Res = []
    Res_mask = []
    start_time = datetime.now()
    for name in names[:mm]:
        i += 1

        print(datetime.now(), i)

        im_file = test_dir + name + '.jpg'
        img = cv2.imread(im_file)
        oh, ow = img.shape[:2]

        with torch.no_grad():
            img = torch.tensor(img)
            res, res_mask = model(img)

        res = res.cpu().numpy()
        # res = res.detach().numpy()
        res_mask = res_mask.cpu()
        res_mask = res_mask.detach().numpy()

        wh = res[:, 2:4] - res[:, :2] + 1

        imgId = int(name)
        m = res.shape[0]

        imgIds = np.zeros((m, 1)) + imgId

        cls = res[:, 5]
        cid = map(lambda x: cls2catId[x], cls)
        cid = list(cid)
        cid = np.array(cid)
        cid = cid.reshape(-1, 1)

        res = np.concatenate((imgIds, res[:, :2], wh, res[:, 4:5], cid), axis=1)
        # Res=np.concatenate([Res,res])
        res = np.round(res, 4)
        Res.append(res)
        Res_mask += map(loadNumpyAnnotations_mask, res[:100], res_mask[:100])

    Res = np.concatenate(Res, axis=0)

    Ann = loadNumpyAnnotations(Res)
    print('==================================', mm, datetime.now() - start_time)
    # with codecs.open('Mask_Rcnn_bbox_ohem_256_gpu2.json', 'w', 'ascii') as f:
    #     json.dump(Ann, f)
    # with codecs.open('Mask_Rcnn_segm_ohem_256_gpu2.json', 'w', 'ascii') as f:
    #     json.dump(Res_mask, f)
    # eval_coco_box.eval('Mask_Rcnn_bbox_ohem_256_gpu2.json', mm)
    # eval_coco_segm.eval('Mask_Rcnn_segm_ohem_256_gpu2.json', mm)

    with codecs.open('py_Mask_Rcnn_bbox.json', 'w', 'ascii') as f:
        json.dump(Ann, f)
    with codecs.open('py_Mask_Rcnn_segm.json', 'w', 'ascii') as f:
        json.dump(Res_mask, f)
    print(mm)
    eval_coco_box.eval('py_Mask_Rcnn_bbox.json', mm)
    eval_coco_segm.eval('py_Mask_Rcnn_segm.json', mm)

    print('==========', datetime.now() - start_time)


if __name__ == "__main__":
    Mean = [123.68, 116.78, 103.94]
    # Mean = [102.9801, 115.9465, 122.7717][::-1]
    config = Config(False, Mean, None, lr=0.00125, weight_decay=0.0001, num_cls=80,img_max=1333, img_min=800,
                    anchor_scales=[[32], [64], [128], [256], [512]],
                    anchor_ratios=[[0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2]], fast_n_sample=512,
                    roi_min_size=[4, 8, 16, 32, 64],
                    roi_train_pre_nms=12000,
                    roi_train_post_nms=2000,
                    roi_test_pre_nms=6000,
                    roi_test_post_nms=1000, )
    model = Mask_Rcnn
    model_file = '/home/zhai/PycharmProjects/Demo35/cascade_rcnn/train_one_GPU/models/Mask_Rcnn_90000_1.pth'

    test(model, config, model_file)
