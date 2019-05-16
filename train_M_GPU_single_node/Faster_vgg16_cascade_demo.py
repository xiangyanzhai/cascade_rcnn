# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

if __name__ == "__main__":
    os.system(
        'python3  -m torch.distributed.launch --nproc_per_node=2   /home/zhai/PycharmProjects/Demo35/cascade_rcnn/train_M_GPU_single_node/Faster_vgg16_cascade.py')



    pass