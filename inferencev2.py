import argparse
import time

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import torchvision.transforms.transforms as transforms
from PIL import Image

from common import allocate_buffers, inference
from prediction import Prediction
from utils.compare import feature_compare

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
transforms = transforms.Compose([
        transforms.Resize((192, 206)),
        transforms.ToTensor(),
        transforms.Normalize(0.568, 0.389)
    ])
engine_path = 'fingerprint.trt'
IMAGE_ROOT = './data/input/FingerprintIdentification/'


def load_engine_from_onnx(onnx_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, 'rb') as model:
        parser.parse(model.read())
    # network = set_net_batch(network)
    builder.max_batch_size = 1
    builder.max_workspace_size = 1 << 30
    network.get_input(0).shape = [1, 1, 192, 206]
    engine = builder.build_cuda_engine(network)
    print('load engine successfully!')
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())

    return engine


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="tensorrt inference")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_path1", type=str, default="image/771.BMP")
    parser.add_argument("--image_path2", type=str, default="image/1006.BMP")
    # parser.add_argument("--engine_path", type=str, default="./fingerprint.trt")
    parser.add_argument("--onnx_path", type=str, default="fingerprint.onnx")
    args = parser.parse_args()

    batch_size = args.batch_size
    onnx_path = args.onnx_path
    engine = load_engine_from_onnx(onnx_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    # image 1
    img1 = Image.open(IMAGE_ROOT + args.image_path1).convert('L')
    im1 = transforms(img1).unsqueeze(0)
    im1 = im1.numpy()
    inputs[0].host = im1
    # inference
    t1 = time.time()
    trt_outputs1 = inference(context, bindings, inputs, outputs, stream)[0]
    print('image1 cost time {}'.format(time.time()-t1))
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    # image2
    img2 = Image.open(IMAGE_ROOT + args.image_path2).convert('L')
    im2 = transforms(img2).unsqueeze(0)
    im2 = im2.numpy()
    inputs[0].host = im2
    # inference
    t2 = time.time()
    trt_outputs2 = inference(context, bindings, inputs, outputs, stream)[0]
    print('image2 cost time {}'.format(time.time() - t2))

    res1 = feature_compare(trt_outputs1, trt_outputs2)
    print(res1)
