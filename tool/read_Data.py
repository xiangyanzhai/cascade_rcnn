# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2
from sklearn.externals import joblib
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class Read_Data(Dataset):
    def __init__(self, config):
        self.img_paths = []
        for i in config.files[0]:
            self.img_paths += joblib.load(i)
        self.Bboxes = []
        for i in config.files[1]:
            self.Bboxes += joblib.load(i)
        self.config = config

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        bboxes = self.Bboxes[index].copy()
        bboxes = bboxes[:, :5]
        H, W = img.shape[:2]
        if bboxes.shape[0]==0:  #当图片中没有目标时，填充一个bboxes，训练时在get_loss中再过滤掉label<0的，只保留label>=0
            bboxes = np.array([[W / 4., H / 4., W * 3 / 4., H * 3 / 4., -2]], dtype=np.float32)

        if np.random.random() > 0.5:
            img = img[:, ::-1].copy()
            bboxes = self.bboxes_left_right(bboxes, img.shape[1])

        img = torch.tensor(img)
        bboxes = torch.tensor(bboxes)
        return img, bboxes, bboxes.shape[0], img.shape[0], img.shape[1]

    def bboxes_left_right(self, bboxes, w):
        bboxes[:, 0], bboxes[:, 2] = w - 1. - bboxes[:, 2], w - 1. - bboxes[:, 0]
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
    return batch
    # # imgs, bboxes, num_bbox, H, W = zip(*batch)
    # m = len(batch)
    # num_bbox = []
    # H = []
    # W = []
    # for i in range(m):
    #     num_bbox.append(batch[i][-3])
    #     H.append(batch[i][-2])
    #     W.append(batch[i][-1])
    # max_num_b = max(num_bbox)
    # max_H = max(H)
    # max_W = max(W)
    #
    # new_img = np.zeros((m, max_H, max_W, 3), dtype=np.uint8)
    # new_bboxes = np.zeros((m, max_num_b, 5), dtype=np.float32)
    # for i in range(m):
    #     new_img[i][:H[i], :W[i]] = batch[i][0]
    #     new_bboxes[i][:num_bbox[i]] = batch[i][1]
    #
    # new_img = torch.from_numpy(new_img)
    # new_bboxes = torch.from_numpy(new_bboxes)
    #
    # num_bbox = torch.tensor(num_bbox)
    # H = torch.tensor(H)
    # W = torch.tensor(W)
    #
    # return new_img, new_bboxes, num_bbox, H, W
    #


if __name__ == "__main__":
    from datetime import datetime
    from py_Faster_tool.tool.config import Config

    Mean = [123.68, 116.78, 103.94]
    path = '/home/zhai/PycharmProjects/Demo35/py_Faster_tool/data_preprocess/'
    Bboxes = [path + 'Bboxes_07.pkl', path + 'Bboxes_12.pkl']
    img_paths = [path + 'img_paths_07.pkl', path + 'img_paths_12.pkl']
    files = [img_paths, Bboxes]
    config = Config(True, Mean, files, lr=0.00125, img_max=1000, img_min=600)
    dataset = Read_Data(config)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=16, collate_fn=lambda x: x)
    c = 0
    for i in range(2):
        for x in dataloader:
            print(len(x))
            img, bboxes = x[0][:2]
            c += 1
            print(datetime.now(), c)
            draw_gt(img.numpy(), bboxes.numpy())
