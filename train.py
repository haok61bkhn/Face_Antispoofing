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
# from torchsummary import summary
from efficientnet_pytorch import EfficientNet

def train(args):
    # gpu init
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # log init
    save_dir = os.path.join(args.save_dir, datetime.now().date().strftime('%Y%m%d'))
    if not os.path.exists(save_dir):
        #raise NameError('model dir exists!')
        os.makedirs(save_dir)
    logging = init_log(save_dir)
    _print = logging.info
    # summary(net.to(config.device), (3,112,112))
    #define tranform
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    net = EfficientNet.from_name('efficientnet-b0', num_classes=2)

    # validation dataset
    trainset = ANTI(train_root="/mnt/sda3/data/FASD", file_list = "train.txt", transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 2,
                                             shuffle=True, num_workers=8, drop_last=False)

    # define optimizers for different layer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD([
        {'params': net.parameters(), 'weight_decay': 5e-4},
    ], lr=0.001, momentum=0.9, nesterov=True)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones= [6, 10, 30], gamma=0.1)
    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)

    total_iters = 1
    vis = Visualizer(env= "effiction")

    for epoch in range(1, args.total_epoch + 1):
        exp_lr_scheduler.step()
        _print('Train Epoch: {}/{} ...'.format(epoch, args.total_epoch))
        net.train()
        since = time.time()
        for data in trainloader:
            img, label = data[0].to(device), data[1].to(device)
            optimizer_ft.zero_grad()
            raw_logits = net(img)
            total_loss = criterion(raw_logits, label)
            total_loss.backward()
            optimizer_ft.step()
            # print train information
            if total_iters % 200 == 0:
                # current training accuracy
                _, predict = torch.max(raw_logits.data, 1)
                total = label.size(0)
                correct = (np.array(predict) == np.array(label.data)).sum()
                time_cur = (time.time() - since) / 100
                since = time.time()
                vis.plot_curves({'softmax loss': total_loss.item()}, iters=total_iters, title='train loss',
                                xlabel='iters', ylabel='train loss')
                vis.plot_curves({'train accuracy': correct / total}, iters=total_iters, title='train accuracy', xlabel='iters',
                                ylabel='train accuracy')

                print("Iters: {:0>6d}/[{:0>2d}], loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".format(total_iters, epoch, total_loss.item(), correct/total, time_cur, exp_lr_scheduler.get_lr()[0]))

            # save model
            if total_iters % args.save_freq == 0:
                msg = 'Saving checkpoint: {}'.format(total_iters)
                _print(msg)
                if multi_gpus:
                    net_state_dict = net.module.state_dict()
                else:
                    net_state_dict = net.state_dict()
                   
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                torch.save({
                    'iters': total_iters,
                    'net_state_dict': net_state_dict},
                    os.path.join(save_dir, 'Iter_%06d_net.ckpt' % total_iters))

            # test accuracy
            if total_iters % args.test_freq == 0 and args.has_test:
                # test model on lfw
                net.eval()
                _print('LFW Ave Accuracy: {:.4f}'.format(np.mean(lfw_accs) * 100))

                net.train()
            total_iters += 1
    print('finishing training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--total_epoch', type=int, default=300, help='total epochs')
    parser.add_argument('--save_freq', type=int, default=5000, help='save frequency')
    parser.add_argument('--test_freq', type=int, default=5000, help='test frequency')
    parser.add_argument('--net_path', type=str, default='weights/MNASMOBILE20190221_023524/Iter_045000_net.ckpt', help='resume model')
    parser.add_argument('--save_dir', type=str, default='./weights', help='model save dir')
    parser.add_argument('--gpus', type=str, default='0', help='model prefix')

    args = parser.parse_args()

    train(args)

