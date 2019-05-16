# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2
from sklearn.externals import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as maskUtils
from torchvision import transforms
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa


def decode_mask(x):
    return maskUtils(x)


def py_decode_mask(counts, mask_h, mask_w):
    mask = map(lambda x: maskUtils.decode({'size': [mask_h, mask_w], 'counts': x}), counts)
    mask = list(mask)
    mask = np.array(mask)
    mask = mask.astype(np.uint8)
    return mask


seq = iaa.Sequential([
    iaa.Affine(
        rotate=[-45, 45], cval=127,
    )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
])


def rotate(img, bboxes, masks):
    o_img = img
    o_bboxes = bboxes
    o_masks = masks
    seq_det = seq.to_deterministic()

    bbox = [ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=l) for x1, y1, x2, y2, l in list(bboxes[:, :5])]
    bbs = ia.BoundingBoxesOnImage(bbox, shape=img.shape)

    image_aug = seq_det.augment_images([img])
    image_aug = image_aug[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    masks = np.transpose(masks, [1, 2, 0])
    masks = seq_det.augment_images([masks])[0]
    masks = np.transpose(masks, [2, 0, 1])

    bbs = bbs_aug.bounding_boxes
    bbs = [[t.x1, t.y1, t.x2, t.y2, t.label] for t in bbs]
    bbs = np.array(bbs)
    h, w = img.shape[:2]
    bboxes = bbs.copy()
    bboxes[:, slice(0, 4, 2)] = np.clip(bboxes[:, slice(0, 4, 2)], 0, w)
    bboxes[:, slice(1, 4, 2)] = np.clip(bboxes[:, slice(1, 4, 2)], 0, h)
    inds = filter_bbox(bboxes, bbs)
    if inds.sum() == 0:
        return o_img, o_bboxes, o_masks

    bboxes = bboxes[inds]
    masks = masks[inds]
    masks[masks != 1] = 0

    bboxes = list(map(adjust_xyxy, bboxes, masks))
    bboxes = np.array(bboxes)
    bboxes = bboxes.astype(np.float32)

    return image_aug, bboxes, masks


def filter_bbox(bboxes, bbs):
    areas1 = (bbs[:, 2] - bbs[:, 0]) * (bbs[:, 3] - bbs[:, 1])
    areas2 = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    inds = (areas2 / areas1) >= 0.7
    return inds


def adjust_xyxy(bbs, mask):
    bbs = bbs.astype(np.int32)
    x1, y1, x2, y2, cls = bbs
    inds = np.where(mask[y1:y2, x1:x2] == 1)
    if len(inds[0])==0:
        return  x1, y1, x2, y2, -2
    y1, y2 = y1 + min(inds[0]), y1 + max(inds[0])
    x1, x2 = x1 + min(inds[1]), x1 + max(inds[1])
    return x1, y1, x2, y2, cls


def img_scale_bboxes_masks(img, bboxes, masks, config):
    h, w = img.shape[:2]
    t_img = np.zeros((h, w, 3), dtype=np.uint8) + np.array([104, 117, 124])
    t_masks = np.zeros((h, w, masks.shape[0]), dtype=np.uint8)
    scale = np.random.choice(config.scale) ** 0.5
    n_h = int(h * scale)
    n_w = int(w * scale)
    y, x = np.random.choice(int(h - n_h)), np.random.choice(int(w - n_w))
    y = min(y, h - n_h)
    x = min(x, w - n_w)

    bboxes[..., :4] = bboxes[..., :4] * scale + np.array([x, y, x, y])
    img = cv2.resize(img, (n_w, n_h))
    masks = np.transpose(masks, (1, 2, 0))
    masks = cv2.resize(masks, (n_w, n_h))
    t_img[y:y + n_h, x:x + n_w] = img
    if len(masks.shape) == 2:
        masks = masks[..., None]
    t_masks[y:y + n_h, x:x + n_w] = masks
    t_masks = np.transpose(t_masks, (2, 0, 1))

    t_img = t_img.astype(np.uint8)
    t_masks = t_masks.astype(np.uint8)

    return t_img, bboxes, t_masks


def crop_img_bboxes_masks(img, bboxes, masks, config):
    h, w = img.shape[:2]
    ori_img = img
    ori_bboxes = bboxes

    jitter = np.random.choice(config.jitter_ratio)
    a = int(h * jitter)
    b = int(w * jitter)
    h1 = np.random.randint(a)
    h2 = np.random.randint(a - h1)
    w1 = np.random.randint(b)
    w2 = np.random.randint(b - w1)

    h2 = h - h2
    w2 = w - w2
    img = img[h1:h2, w1:w2]

    x1 = np.maximum(w1, bboxes[:, 0:1])
    y1 = np.maximum(h1, bboxes[:, 1:2])
    x2 = np.minimum(w2 - 1, bboxes[:, 2:3])
    y2 = np.minimum(h2 - 1, bboxes[:, 3:4])

    x1 = x1 - w1
    y1 = y1 - h1
    x2 = x2 - w1
    y2 = y2 - h1

    areas1 = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    w = np.maximum(0., x2 - x1)
    h = np.maximum(0., y2 - y1)
    areas2 = w * h
    areas2 = areas2.ravel()

    inds = (areas2 / areas1) >= config.crop_iou

    bboxes = np.concatenate([x1, y1, x2, y2, bboxes[:, 4:5]], axis=1)
    bboxes = bboxes[inds]

    if bboxes.shape[0] == 0 or np.random.random() < config.keep_ratio:
        return ori_img, ori_bboxes, masks
    return img, bboxes, masks[inds, h1:h2, w1:w2]


class Read_Data(Dataset):
    def __init__(self, config):
        self.img_paths = []
        for i in config.files[0]:
            self.img_paths += joblib.load(i)
        self.Bboxes = []
        for i in config.files[1]:
            self.Bboxes += joblib.load(i)
        self.Counts = []
        for i in config.files[2]:
            self.Counts += joblib.load(i)

        self.transform = transforms.Compose([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)])
        self.config = config

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        bboxes = self.Bboxes[index].copy()

        bboxes = bboxes[:, :5]
        H, W = img.shape[:2]
        counts = self.Counts[index]

        if len(counts) > 0:
            masks = py_decode_mask(counts, H, W)
        else:
            masks = np.ones((1, H, W), dtype=np.uint8)  # 当图片中没有目标时，填充一个bboxes，训练时在get_loss中再过滤掉label<0的，只保留label>=0
            bboxes = np.array([[W / 4., H / 4., W * 3 / 4., H * 3 / 4., -2]], dtype=np.float32)
        #
        # if np.random.random() > 0.5:
        #     img = Image.fromarray(img[..., ::-1])
        #     img = self.transform(img)
        #     img = np.array(img)
        #     img = np.ascontiguousarray(img[..., ::-1])
        #
        # if np.random.random() > 0.5:
        #     img, bboxes, masks = rotate(img, bboxes, masks)

        if np.random.random() > 0.5:
            img = img[:, ::-1].copy()
            bboxes = self.bboxes_left_right(bboxes, W)
            masks = masks[..., ::-1].copy()

        # if np.random.random() > 0.5:
        #     img = img[::-1].copy()
        #     bboxes = self.bboxes_up_down(bboxes, H)
        #     masks = masks[:, ::-1].copy()
        #
        # if bboxes.shape[0] > 0:
        #     img, bboxes, masks = crop_img_bboxes_masks(img, bboxes, masks, self.config)
        #
        # if np.random.random() > 0.5:
        #     img, bboxes, masks = img_scale_bboxes_masks(img, bboxes, masks, self.config)

        img = torch.tensor(img)
        bboxes = torch.tensor(bboxes)
        masks = torch.tensor(masks)

        return img, bboxes, bboxes.shape[0], img.shape[0], img.shape[1], masks

    def bboxes_left_right(self, bboxes, w):
        bboxes[:, 0], bboxes[:, 2] = w - 1. - bboxes[:, 2], w - 1. - bboxes[:, 0]
        return bboxes

    def bboxes_up_down(self, bboxes, h):
        bboxes[:, 1], bboxes[:, 3] = h - 1. - bboxes[:, 3], h - 1. - bboxes[:, 1]
        return bboxes


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


def draw_mask(im, gt, mask):
    print(im.shape, mask.shape)
    im = im.astype(np.uint8)
    boxes = gt.astype(np.int32)

    c = 0
    for box in boxes:
        # print(box)

        t = mask[c, :, :]

        inds = np.where(t == 1)
        im[inds] = np.array([0, 0, 255], )
        x1, y1, x2, y2 = box[:4]

        im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255))
        # print(box[-1])
        c += 1
    im = im.astype(np.uint8)
    cv2.imshow('a', im)
    cv2.waitKey(2000)
    return im


if __name__ == "__main__":
    from datetime import datetime
    from cascade_rcnn.tool.config import Config

    Mean = [123.68, 116.78, 103.94]
    path = '/home/zhai/PycharmProjects/Demo35/pytorch_Faster_tool/data_preprocess/'
    Bboxes = [path + 'coco_bboxes_2017.pkl']
    img_paths = [path + 'coco_imgpaths_2017']
    masks = [path + 'coco_mask_2017.pkl']
    files = [img_paths, Bboxes, masks]
    config = Config(True, Mean, files, lr=0.00125, img_max=1000, img_min=600)
    dataset = Read_Data(config)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=16, collate_fn=lambda x: x)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=func,
                            shuffle=True, drop_last=True, pin_memory=False, num_workers=1)
    c = 0
    for i in range(2):
        for imgs, bboxes, num_b, num_H, num_W, masks in dataloader:
            # imgs, bboxes, num_b, num_H, num_W, masks = x
            c += 1
            print(datetime.now(), c, masks.shape, imgs.shape)
            # print(img.shape, bboxes.shape, masks.shape)
            for j in range(imgs.shape[0]):
                x = imgs[j]
                H = num_H[j]
                W = num_W[j]
                b = num_b[j]
                mask = masks[j]
                bb = bboxes[j]
                x = x.view(-1)[:H * W * 3].view(H, W, 3)
                bb = bb[:b]
                mask = mask.view(-1)[:b * H * W].view(b, H, W)
                draw_mask(x.numpy(), bb.numpy(), mask.numpy())
