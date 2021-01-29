import argparse
import time

import tensorrt as trt
from PIL import Image
import torchvision.transforms.transforms as transforms
import numpy as np
import pycuda.driver as cuda

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n{}\nDevice:\n{}".format(self.host, self.device)

    def __repr__(self):
        return self.__repr__()


def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # append the device buffer to device bindings
        bindings.append(int(device_mem))
        # append to the appropriate list
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem)
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def inference(engine, batch_size, im, output_shape):
    context = engine.create_execution_context()

    inputs, outputs, bindings, stream = allocate_buffers(engine)

    output = np.empty(output_shape, dtype=np.float)

    # 分配内存
    d_input = cuda.mem_alloc(1 * im.size * im.dtype.itemsize)
    d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
    bindings = [int(d_input), int(d_output)]

    # pycuda操作缓冲区
    stream = cuda.Stream()
    # 将输入数据放入device
    cuda.memcpy_htod_async(d_input, im, stream)

    start = time.time()
    # 执行模型
    context.execute_async(batch_size, bindings, stream.handle, None)
    # 将预测结果从缓冲区取出
    cuda.memcpy_dtoh_async(output, d_output, stream)
    end = time.time()

    # 线程同步
    stream.synchronize()

    print(output)
    print(end-start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="tensorrt inference")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_path", type=str, default="./14.BMP")
    parser.add_argument("--engine_path", type=str, default="./fingerprint.trt")
    args = parser.parse_args()

    batch_size = args.batch_size
    engine_path = args.engine_path
    engine = load_engine(engine_path)
    img = Image.open(args.image_path).convert('L')
    transforms = transforms.Compose([
        transforms.Resize((192, 206)),
        transforms.ToTensor(),
        transforms.Normalize(0.568, 0.389)
    ])
    im = transforms(img).unsqueeze(0)
    im = im.numpy()
    output_shape = [1, 1]
    inference(engine, batch_size, im, output_shape)