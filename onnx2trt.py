#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/01/19
"""

import tensorrt as trt
import pycuda.autoinit

onnx_path = './fingerprint.onnx'
engine_path = 'fingerprint.trt'

TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def set_net_batch(network):
    shape = list(network.get_input(0).shape)
    shape[0] = 1
    network.get_input(0).shape = shape
    return network


def onnx2engine():
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, 'rb') as model:
        parser.parse(model.read())
    builder.max_batch_size = 1
    builder.max_workspace_size = 1 << 30
    network.get_input(0).shape = [1, 1, 192, 206]
    # print(network)
    engine = builder.build_cuda_engine(network)
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    print('success')


if __name__ == '__main__':
    onnx2engine()
