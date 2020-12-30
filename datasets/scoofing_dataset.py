#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/12/30
"""
import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from path import DATA_PATH


class SocoFing(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        self.dataset_root_path = os.path.join(DATA_PATH, 'SocoFing')
        self.dataset_altered_path = os.path.join(self.dataset_root_path, 'Altered')
        self.real_image_path = os.path.join(self.dataset_root_path, 'Real')
        self.altered_easy_path = os.path.join(self.dataset_altered_path, 'Altered-Easy')
        self.altered_hard_path = os.path.join(self.dataset_altered_path, 'Altered-Hard')
        self.altered_medium_path = os.path.join(self.dataset_altered_path, 'Altered-Medium')
        self.train_images = []
        self.train_labels = []
        self.valid_images = []
        self.valid_labels = []
        self.test_images = []
        self.test_labels = []
        self.real_image_num = 0
        self.easy_image_num = 0
        self.medium_image_num = 0
        self.hard_image_num = 0
        self.transforms = transforms.Compose([
            transforms.Resize((192, 206)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(0.568, 0.389)
        ])

        self.process_data()

    def __getitem__(self, index):
        if self.mode == 'test':
            image = self.test_labels[index]
            label = self.test_labels[index]
        elif self.mode == 'valid':
            image = self.valid_images[index]
            label = self.valid_labels[index]
        else:
            image = self.train_images[index]
            label = self.train_labels[index]
        im = Image.open(image).convert('L')
        im = self.transforms(im)
        return im, label

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_images)
        elif self.mode == 'valid':
            return len(self.valid_images)
        else:
            return len(self.train_images)

    def process_data(self):
        real_images = os.listdir(self.real_image_path)
        easy_images = os.listdir(self.altered_easy_path)
        medium_images = os.listdir(self.altered_medium_path)
        hard_images = os.listdir(self.altered_hard_path)
        self.real_image_num = len(real_images)
        self.easy_image_num = len(easy_images)
        self.medium_image_num = len(medium_images)
        self.hard_image_num = len(hard_images)

        # real image
        _train_images, _train_labels, _test_images, _test_labels, _valid_images, _valid_labels = \
            self._get_train_data(self.real_image_num, self.real_image_path, real_images)
        self.train_images += _train_images
        self.train_labels += _train_labels
        self.test_images += _test_images
        self.test_labels += _test_labels
        self.valid_images += _valid_images
        self.valid_labels += _valid_labels

        # altered easy
        _train_images, _train_labels, _test_images, _test_labels, _valid_images, _valid_labels = \
            self._get_train_data(self.easy_image_num, self.altered_easy_path, easy_images)
        self.train_images += _train_images
        self.train_labels += _train_labels
        self.test_images += _test_images
        self.test_labels += _test_labels
        self.valid_images += _valid_images
        self.valid_labels += _valid_labels

        # altered medium
        _train_images, _train_labels, _test_images, _test_labels, _valid_images, _valid_labels = \
            self._get_train_data(self.medium_image_num, self.altered_medium_path, medium_images)
        self.train_images += _train_images
        self.train_labels += _train_labels
        self.test_images += _test_images
        self.test_labels += _test_labels
        self.valid_images += _valid_images
        self.valid_labels += _valid_labels

        # altered hard
        _train_images, _train_labels, _test_images, _test_labels, _valid_images, _valid_labels = \
            self._get_train_data(self.hard_image_num, self.altered_hard_path, hard_images)
        self.train_images += _train_images
        self.train_labels += _train_labels
        self.test_images += _test_images
        self.test_labels += _test_labels
        self.valid_images += _valid_images
        self.valid_labels += _valid_labels

    def _get_train_data(self, image_nums, image_path, images):
        # train : test :valid 7 : 2 : 1
        valid_test_num = int(0.3 * image_nums)
        rd_valid_test_nums = random.sample(range(image_nums), valid_test_num)
        valid_num = int(0.1 * image_nums)
        valid_nums = rd_valid_test_nums[:valid_num]
        test_nums = rd_valid_test_nums[valid_num:]
        _valid_images = []
        _valid_labels = []
        _test_images = []
        _test_labels = []
        _train_images = []
        _train_labels = []
        for i, image in enumerate(images):
            full_image = os.path.join(image_path, image)
            label = int(image.split('__')[0]) - 1
            if i in valid_nums:
                _valid_images.append(full_image)
                _valid_labels.append(label)
            elif i in test_nums:
                _test_images.append(full_image)
                _test_labels.append(label)
            else:
                _train_images.append(full_image)
                _train_labels.append(label)
        return _train_images, _train_labels, _test_images, _test_labels, _valid_images, _valid_labels



