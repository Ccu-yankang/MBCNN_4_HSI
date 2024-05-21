import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import time
import cv2
import h5py
import random
import warnings
import argparse
import numpy as np
import net as net
import evaluation as eva


dir = 'D:/workspace/MBCNN/dataset/HS-SOD/'
dir_txt = os.path.join('./net/', "information.txt")
channel_path = './net_test/channel.txt'
flag=True
with open(dir_txt, "r") as f:
    while(flag):
        text=f.readline()
        if(text=="test_list:\n"):
            image_test_list_str=f.read()
            flag=False

image_test_list=image_test_list_str.split("\n")
image_test_list.pop()

use_cuda=torch.cuda.is_available()
nChannels=64
sum_test_nLabels = 0
sum_test_loss = 0
image_test_num=len(image_test_list)

loss_fn = torch.nn.CrossEntropyLoss()
#加载模型
model=net.TestNet3()
model.load_state_dict(torch.load("./net/MBCNN_epoch_10.pkl"))

channels = net.Sample(3,channel_path)
channels.calc_sample()

for index, image_name in enumerate(image_test_list, 1):
    # load image
    image = h5py.File(os.path.join('D:/workspace/MBCNN/dataset/HS-SOD/', "hyperspectral", image_name))
    image_data = np.expand_dims(image["hypercube"][()].astype('float32'), axis=0)  # shape: [1, 81, 1024, 768]
    data = image_data / np.max(image_data)
    # data = torch.from_numpy(data)
    # if use_cuda:
    #     data = data.cuda()
    # data = Variable(data)


    # forward
    a = channels.get_sample()
    output = model(data, a)  # shape: [args.nChannels, 1024, 768]
    vote=net.Vote(output)
    saliency_map=vote.vote(0)

    # output = output.permute(0, 1, 3, 2)
    # output=torch.max(output,dim=1)

    # output=1-output[1]

    # display
    # img = output.data.cpu().numpy()
    img = saliency_map[1] * 255
    cv2.imshow("image", img.astype(float))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output = output.contiguous().view(-1, 1)  # shape: [1024*768, args.nChannels]
    # output = output.permute(1, 0).contiguous().view(-1, 1)  # shape: [1024*768, args.nChannels]


    # _, target = torch.max(output, 1)
    # target = target.data.cpu().numpy()
    # nLabels = len(np.unique(target))

    #ground_truth
    image_num = image_name.split('.')
    img = image_num[0] + '.jpg'
    ground_truth_path = os.path.join(dir, 'ground_truth', img)
    ground_truch = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    ground_truch = cv2.threshold(ground_truch, 0.49, 1, cv2.THRESH_BINARY)
    ground_truch=ground_truch[1]
    # ground_truch=np.array(ground_truch)
    # ground_truch=torch.tensor(ground_truch)
    ground_truch = torch.from_numpy(ground_truch)
    ground_truch = ground_truch.contiguous().view(1, -1).float()
    # print(ground_truch[0][torch.argmax(ground_truch)])

    #display
    # ground_truch = ground_truch.numpy()

    # filepath=image_name.split(".")
    # filepath=filepath[0]+".jpg"
    # filepath=os.path.join("D:\workspace\MBCNN\output",filepath)
    # cv2.imwrite(filepath,img)

    # print(filepath)
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # ground_truch = torch.from_numpy(ground_truch)

    # loss

    # if use_cuda:
    #     ground_truch = ground_truch.cuda()
    # ground_truch = Variable(ground_truch)
    # loss = loss_fn(output, ground_truch[0].long())

    #evaluation
    # P=eva.precision(output,ground_truch[0])
    # R=eva.recall(output,ground_truch[0])
    # F=eva.F_measure(P,R)
    # print(F)

    # sum_test_nLabels += nLabels
    # sum_test_loss += loss.item()



print("[Test] Avg nLabels: %.2f, Avg Loss: %.4f\n" %
      (sum_test_nLabels / image_test_num, sum_test_loss / image_test_num))