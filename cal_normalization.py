#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/12/25
"""
import os
import numpy as np
import cv2

image_path = './data/input/FingerprintIdentification/image/'


def cal():
    images = os.listdir(image_path)
    mean = 0
    std = 0
    for image in images:
        im = cv2.imread(image_path + image, cv2.IMREAD_GRAYSCALE)
        im = im / 255.0
        mean += np.mean(im)
        std += np.std(im)
        # cv2.imshow('im', im)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # break
    mean = mean / len(images)
    std = std / len(images)
    print(mean, std)
    # mean  0.5682917131747312
    # std   0.38947670385076305


if __name__ == '__main__':
    cal()
