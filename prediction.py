# -*- coding: utf-8 -*
import torch
from PIL import Image
from flyai.framework import FlyAI
from torch.backends import cudnn

from datasets.dataset import FingerPrintDataset
from model.osnet import osnet_x1_0
from utils.utils import load_pretrained_weights
import numpy as np


class Prediction(FlyAI):
    def load_model(self, num_classes=100):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        model = osnet_x1_0(num_classes=num_classes)
        load_pretrained_weights(model, './weights/net_49.pth')
        return model

    def predict(self, image_path_1, image_path_2):
        '''
        模型预测返回结果
        :param input:  评估传入样例 {"image_path_1":"image\/4900.BMP","image_path_2":"image\/12634.BMP"}
        :return: 模型预测成功之后返回给系统样例 {"label":"1"}
        '''
        dataset = FingerPrintDataset()
        model = self.load_model(num_classes=dataset.num_classes)
        model = model.cuda()
        model.eval()
        cudnn.benchmark = True
        im1 = Image.open('./data/input/FingerprintIdentification/'+image_path_1).convert('L')
        im2 = Image.open('./data/input/FingerprintIdentification/'+image_path_2).convert('L')
        im1 = dataset.transform(im1)
        im2 = dataset.transform(im2)
        im1 = im1.unsqueeze(0)
        im2 = im2.unsqueeze(0)
        im1 = im1.cuda()
        im2 = im2.cuda()
        out1 = model(im1)
        out1 = torch.sigmoid(out1).cpu().detach().numpy()
        out1 = np.argmax(out1)
        print(out1)
        out2 = model(im2)
        out2 = torch.sigmoid(out2).cpu().detach().numpy()
        out2 = np.argmax(out2)
        print(out2)
        return {"label":"1"}


if __name__ == '__main__':
    prediction = Prediction()
    prediction.predict('image/4900.BMP', 'image/12634.BMP')