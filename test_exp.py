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

branch_num = 3
branch_layer=5
sample_channels=54
branch_net='End_to_End_L'+str(branch_layer)
net_name=branch_net+'_B'+str(branch_num)
model_file=net_name+'_epoch_10.pkl'

dataset_dir = './dataset/HS-SOD/'
base_net_path='./net/'
net_path = os.path.join(base_net_path,net_name+'/')
model_path=os.path.join(net_path,model_file)
information_file='information.txt'
dir_txt = os.path.join(net_path, information_file)
channel_file = "channel.txt"
channel_path = os.path.join(net_path,channel_file)
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
model.load_state_dict(torch.load(model_path))

channels = net.Sample(branch_num,channel_path)
channels.calc_sample()

for index, image_name in enumerate(image_test_list, 1):
    # load image
    image = h5py.File(os.path.join(dataset_dir, "hyperspectral", image_name))
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
    ground_truth_path = os.path.join(dataset_dir, 'ground_truth', img)
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