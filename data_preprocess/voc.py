# !/usr/bin/python
# -*- coding:utf-8 -*-

import os
import numpy as np

from lxml import etree
from sklearn.externals import joblib

Label = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

Label_dict = {}

c = 0
for i in Label:
    Label_dict[i] = c
    Label_dict[c] = i
    c += 1
print(Label_dict)
train_dir = r'/home/zhai/PycharmProjects/Demo35/dataset/voc/VOCdevkit/VOC2007/JPEGImages/'
names = os.listdir(train_dir)
names = [name.split('.')[0] for name in names]
names = sorted(names)
ann_path = '/home/zhai/PycharmProjects/Demo35/dataset/voc/VOCdevkit/VOC2007/Annotations/'
c = 0
img_paths = []
Bboxes = []
# np.random.seed(50)
# np.random.shuffle(names)
for name in names:
    c += 1
    print(c)
    htm = etree.parse(ann_path + name + '.xml')
    bboxes = []
    for obj in htm.xpath('//object'):
        cls = obj.xpath('name')[0].xpath('string(.)').strip()
        difficult = obj.xpath('difficult')[0].xpath('string(.)').strip()
        bbox = obj.xpath('bndbox')[0]
        x1 = bbox.xpath('xmin')[0].xpath('string(.)').strip()
        x2 = bbox.xpath('xmax')[0].xpath('string(.)').strip()
        y1 = bbox.xpath('ymin')[0].xpath('string(.)').strip()
        y2 = bbox.xpath('ymax')[0].xpath('string(.)').strip()
        difficult = int(difficult)
        x1 = float(x1) - 1
        x2 = float(x2) - 1
        y1 = float(y1) - 1
        y2 = float(y2) - 1
        if difficult == 1:
            continue
        bboxes.append([x1, y1, x2, y2, Label_dict[cls]])

    if len(bboxes) == 0:
        print('********************************************', c)
        continue
    bboxes = np.array(bboxes)
    bboxes = bboxes.astype(np.float32)
    img_paths.append(train_dir + name + '.jpg')

    Bboxes.append(bboxes)
joblib.dump(img_paths, 'img_paths_07.pkl')
joblib.dump(Bboxes, 'Bboxes_07.pkl')
