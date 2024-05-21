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
import torch.nn.functional as F
import net as net


#变量:
#   分支数
#   分支网络层数与网络结构
#   投票规则
#   损失函数
#   通道抽样层数
if __name__ == "__main__":

    branch_num = 3
    branch_layer=5
    branch_net='End_to_End_L'+str(branch_layer)
    vote_rule=0
    sample_channels=54
    loss_function=''

    # net_name=f'{branch_net}_B{str(branch_num)}'
    net_name=branch_net+'_B'+str(branch_num)
    output_path='./output/'
    base_net_path='./net/'
    net_path = os.path.join(base_net_path,net_name+'/')

    channel_file = "channel.txt"
    channel_path = os.path.join(net_path,channel_file)
    hyperspectral_channels = 81
    information_file='information.txt'
    dataset_dir = './dataset/HS-SOD/'
    model_flag = False

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(net_path):
        os.mkdir(net_path)
        channel_dir = os.path.join(net_path, channel_file)
        lists=list(range(0,hyperspectral_channels))
        with open(channel_dir, 'w+') as f:
            for i in range(branch_num):
                channel_list = random.sample(lists, sample_channels)
                channel_list.sort()
                f.write(str(channel_list)+"\n")
                #f.write(str(int(random.random()*100)) + "\n")
                #f.write(str(int(random.random()*100)) + "\n")

    image_list = os.listdir(os.path.join(dataset_dir, "hyperspectral"))
    image_list.sort()
    image_train_list = image_list[0::2]
    image_test_list = image_list[1::2]
    image_train_num = len(image_train_list)
    image_test_num = len(image_test_list)

    dir_txt = os.path.join(net_path, information_file)
    with open(dir_txt, "w+") as f:
        image_train_list.sort()
        f.write("train_list:\n")
        for image_name in image_train_list:
            f.write(image_name + "\n")
        image_test_list.sort()
        f.write("\ntest_list:\n")
        for image_name in image_test_list:
            f.write(image_name + "\n")

    channels = net.Sample(branch_num,channel_path)
    channels.calc_sample()

    for epoch in range(10):
        # train model
        random.shuffle(image_train_list)
        for index, image_name in enumerate(image_train_list, 1):
            # load image
            image = h5py.File(os.path.join(dataset_dir, "hyperspectral", image_name))
            image_data = np.expand_dims(image["hypercube"][()].astype('float32'), axis=0)  # shape: [1, 81, 1024, 768]
            data = image_data / np.max(image_data)
            #训练前需要随机抽取通道，torch数据不方便抽取
            # data = torch.from_numpy(data)
            # data = Variable(data)

            # model
            if not model_flag:
                #根据变量更换
                model=net.TestNet3()
                #====================================================================================
                model.train()
                # 根据变量更换
                loss_fn = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.95)
                # ====================================================================================
                model_flag = True

            # forward
            optimizer.zero_grad()
            a=channels.get_sample()
            output = model(data, a)  # shape: [args.nChannels, 1024, 768]
            output=output.permute(0,3,2,1).contiguous().view(branch_num,-1,2)
            # print(output.ndim)

            #ground_truth
            image_num=image_name.split('.')
            img=image_num[0]+'.jpg'
            ground_truth_path = os.path.join(dataset_dir, 'ground_truth',img)
            ground_truch=cv2.imread(ground_truth_path,cv2.IMREAD_GRAYSCALE)
            ground_truch = cv2.threshold(ground_truch, 0.49, 1, cv2.THRESH_BINARY)
            ground_truch=ground_truch[1]
            ground_truch=torch.from_numpy(ground_truch)
            ground_truch=ground_truch.contiguous().view(1,-1).float()

            # loss and backward
            # target = torch.from_numpy(target)
            # target = Variable(target)
            # loss = loss_fn(output, target)

            #ground_truch需要求梯度？
            #loss 只能用一次
            for i in range(branch_num):
                ground_truch=torch.cat((ground_truch,ground_truch))
            ground_truch=Variable(ground_truch)
            loss=loss_fn(output[i],ground_truch[0].long())
            print(loss)
            loss.backward()
            optimizer.step()

        #save net
        # model_name = net_name + "_epoch_%d.pkl" % (epoch + 1)
        model_name = net_name + "_epoch_{}.pkl" .format(str(epoch + 1))
        torch.save(model.state_dict(), os.path.join(net_path, model_name))







