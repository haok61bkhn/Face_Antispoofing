#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: train.py.py
@time: 2018/12/21 17:37
@desc: train script for deep face recognition
'''
import os
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime

from utils import Visualizer
from utils import init_log
from dataset import ANTI
from torch.optim import lr_scheduler
import torch.optim as optim
import time
import numpy as np
import torchvision.transforms as transforms
import argparse
from PIL import Image
import cv2
from imutils.video import VideoStream
from retinaface import  Retinaface_Detector
# from torchsummary import summary
from efficientnet_pytorch import EfficientNet
def test_cam(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    net = EfficientNet.from_name('efficientnet-b0', num_classes=2)
    checkpoint = torch.load("./weights/20200704/Iter_000900_net.ckpt", map_location=lambda storage, loc: storage)
    # print(checkpoint.keys())
    net.load_state_dict(checkpoint['net_state_dict'])
    net.eval()
    net = net.to(device)
    retina = Retinaface_Detector()
    vid = VideoStream("rtsp://admin:meditech123@192.168.100.64:554").start()
    frame = vid.read()
    while True:
        frame = vid.read()
        if frame is None:
            continue
        dicta = retina.align_multi(frame)
        for box in dicta["bboxs"]:
            face =  frame[box[1]:box[3], box[0]: box[2]]
            img = transform(Image.fromarray(face.astype('uint8')))
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            raw_logits = net(img.to(device))
            raw_logits = torch.nn.functional.softmax(raw_logits)
            score, predict = torch.max(raw_logits.data, 1)
            if predict[0]== 1 and score[0] > 0.8:
                print(score, predict)
                smoof_stt = "real"
            else:
                smoof_stt = "smoofing"

            frame = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
            frame = cv2.putText(frame, smoof_stt, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 0, 0), 1, cv2.LINE_AA) 
        cv2.imshow("window_name", frame) 
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--image', type=str, default="imgs/FT720P_id8_s0_90.png", help='total epochs')
    args = parser.parse_args()

    test_cam(args)

