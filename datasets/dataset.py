#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/12/23
"""
import os

import pandas
from torch.utils.data import Dataset
from path import DATA_PATH


class FingerPrintDataset(Dataset):
    def __init__(self, image_path='', label_path=''):
        self.image_path = image_path
        self.label_path = label_path
        self.train_images = []
        self.labels = []
        self.num_classes = 0
        self.sample_nums = 0
        self.label_map = {}

    def __getitem__(self, index):
        return self.train_images[index], self.labels[index]

    def process_data(self):
        # images = os.listdir(self.image_path)
        images_labels = self.read_train_csv()

    def read_train_csv(self):
        csv = 'data/input/FingerprintIdentification/train.csv'
        dataframe = pandas.read_csv(csv)
        image_path1s = dataframe['image_path_1']
        image_path2s = dataframe['image_path_2']
        origin_labels = dataframe['label']
        # print(image_path1s)
        # print('##########################################')
        # print(image_path2s)
        # print('##########################################')
        # print(origin_labels)
        pid = 0
        for image1, image2, label in zip(image_path1s, image_path2s, origin_labels):
            image1, image2 = image1.replace('', 'image/'), image2.replace('', 'image/')
            if image1 not in self.label_map and image2 not in self.label_map:
                # 新的标签
                if label == 1:
                    self.label_map[image1], self.label_map[image2] = pid, pid
                    pid += 1
                else:
                    self.label_map[image1] = pid
                    pid += 1
                    self.label_map[image2] = pid
                    pid += 1
            else:
                pass



