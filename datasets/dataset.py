#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/12/23
"""
import os

import pandas
from PIL import Image
from torch.utils.data import Dataset
from path import DATA_PATH
import torchvision.transforms as transforms


class FingerPrintDataset(Dataset):
    def __init__(self, image_path='', label_path=''):
        self.image_path = os.path.join(os.path.join(DATA_PATH, 'FingerprintIdentification'), 'image')
        self.label_path = label_path
        self.train_images = []
        self.labels = []
        self.num_classes = 0
        self.sample_nums = 0
        self.label_map = {}
        self.transform = transforms.Compose(
            [
                transforms.Resize((192, 206)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(0.568, 0.389)
            ]
        )

        self.process_data()

    def __getitem__(self, index):
        image = self.train_images[index]
        label = self.labels[index]
        im = Image.open(os.path.join(self.image_path, image)).convert('L')
        im = self.transform(im)
        # print(im.shape)
        return im, label

    def process_data(self):
        images = os.listdir(self.image_path)
        self.read_train_csv()
        for image in images:
            self.train_images.append(image)
            self.labels.append(self.label_map[image])

    def read_train_csv(self):
        csv = 'data/input/FingerprintIdentification/train.csv'
        dataframe = pandas.read_csv(csv)
        image_path1s = dataframe['image_path_1']
        image_path2s = dataframe['image_path_2']
        origin_labels = dataframe['label']
        pid = 0
        # 按照分类的方法处理重识别
        # 先处理相同的 后处理不同的
        for image1, image2, label in zip(image_path1s, image_path2s, origin_labels):
            image1, image2 = image1.replace('image/', ''), image2.replace('image/', '')
            if label == 0:
                continue
            if image1 not in self.label_map and image2 not in self.label_map:
                self.label_map[image1], self.label_map[image2] = pid, pid
                for im1, im2, lb in zip(image_path1s, image_path2s, origin_labels):
                    im1, im2 = im1.replace('image/', ''), im2.replace('image/', '')
                    if lb == 0:
                        continue
                    if im1 == image1 and im2 not in self.label_map:
                        self.label_map[im2] = self.label_map[image1]
                    if im2 == image2 and im1 not in self.label_map:
                        self.label_map[im1] = self.label_map[image2]
                pid += 1
        for image1, image2, label in zip(image_path1s, image_path2s, origin_labels):
            image1, image2 = image1.replace('image/', ''), image2.replace('image/', '')
            if label == 1:
                continue
            if image1 not in self.label_map:
                self.label_map[image1] = pid
                pid += 1
            if image2 not in self.label_map:
                self.label_map[image2] = pid
                pid += 1
        self.num_classes = pid

    def __len__(self):
        return len(self.train_images)



