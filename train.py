#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/12/23
"""

import argparse
import os

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.scoofing_dataset import SocoFing
from model.osnet import osnet_x1_0
from path import MODEL_PATH
from utils.utils import load_pretrained_weights, build_optimizer, build_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np


if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=50, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=4, type=int, help="batch size")
args = parser.parse_args()


def train():
    max_epoch = args.EPOCHS
    train_batch_size = args.BATCH
    valid_test_batch_size = 1

    train_dataset = SocoFing(mode='train')
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    valid_dataset = SocoFing(mode='valid')
    valid_loader = DataLoader(valid_dataset, batch_size=valid_test_batch_size)

    test_dataset = SocoFing(mode='test')
    test_loader = DataLoader(test_dataset, batch_size=valid_test_batch_size)

    model = osnet_x1_0(num_classes=600, pretrained=True, loss='softmax', use_gpu=True)
    model = model.cuda()

    optimizer = build_optimizer(model)

    scheduler = build_scheduler(optimizer, lr_scheduler='cosine', max_epoch=max_epoch)
    criterion = nn.CrossEntropyLoss()

    cudnn.benchmark = True
    writer = SummaryWriter(log_dir='./log')
    for epoch in tqdm(max_epoch):
        model.train()
        for index, data in enumerate(train_loader):
            im, label = data
            im = im.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            out = model(im)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            if index % 10 == 0:
                print("Epoch: [{}/{}][{}/{}]  Loss {:.4f}".format(epoch+1, max_epoch, index+1,
                                                                              len(train_loader), loss))
                n_iter = epoch*len(train_loader) + index
                writer.add_scalar('loss', loss, n_iter)
        # save checkpoint
        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), './weights/net_'+str(epoch)+'.pth')

        # valid
        true_count = 0
        for index, data in enumerate(valid_loader):
            model.eval()
            im, label = data
            im = im.cuda()
            out = model(im)
            out = torch.softmax(out, 1).cpu().detach().numpy()
            pred_y = np.argmax(out)
            if pred_y == label:
                true_count += 1
        valid_acc = true_count / len(valid_loader) * 100.0
        writer.add_scalar('valid acc', valid_acc, epoch)

        scheduler.step()
    torch.save(model.state_dict(), './weights/last.pth')
    writer.close()

    # test
    true_count = 0
    for index, data in enumerate(test_loader):
        model.eval()
        im, label = data
        im = im.cuda()
        label = label.cuda()
        out = model(im)
        out = torch.softmax(out, 1).cpu().detach().numpy()
        pred_y = np.argmax(out)
        if pred_y == label:
            true_count += 1
    test_acc = true_count / len(valid_loader) * 100.0
    print('test acc {.2f}%'.format(test_acc))


if __name__ == '__main__':
    train()
