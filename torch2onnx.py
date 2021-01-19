#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/01/19
"""

import torch

from datasets.fingerprint_dataset import FingerPrintDataset
from model.osnet import osnet_x1_0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './weights/net_49.pth'


def torch2onxx():
    onxx_model = "fingerprint.onnx"
    dummy_input = torch.randn(1, 1, 192, 206, requires_grad=True)
    dataset = FingerPrintDataset()
    model = osnet_x1_0(num_classes=dataset.num_classes)
    model.load_state_dict(torch.load(model_path))
    torch.onnx.export(model,
                                dummy_input,
                                onxx_model, verbose=False,
                                training=False, do_constant_folding=True,
                                input_names=['input'],
                                output_names=['output']
                                )

    print("onnx generated successfully!")


if __name__ == '__main__':
    torch2onxx()