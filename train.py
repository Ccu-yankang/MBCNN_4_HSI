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

if __name__ == "__main__":

    if not os.path.exists('./output/'):
        os.mkdir('./output/')
    if not os.path.exists('./net/'):
        os.mkdir('./net/')
        channel_dir = os.path.join('./net/', "channel.txt")
        lists=list(range(1,82))
        with open(channel_dir, 'w+') as f:
            for i in range(2):
                channel_list = random.sample(lists, 54)
                channel_list.sort()
                f.write(str(channel_list)+"\n")
                #f.write(str(int(random.random()*100)) + "\n")
                #f.write(str(int(random.random()*100)) + "\n")


    model_flag = False
    dir = 'D:/workspace/MBCNN/dataset/HS-SOD/'
    nChannels=64

    image_list = os.listdir(os.path.join('./dataset/HS-SOD/', "hyperspectral"))
    image_list.sort()
    image_train_list = image_list[0::2]
    image_test_list = image_list[1::2]
    image_train_num = len(image_train_list)
    image_test_num = len(image_test_list)

    dir_txt = os.path.join('./net/', "information.txt")
    with open(dir_txt, "w+") as f:
        image_train_list.sort()
        f.write("train_list:\n")
        for image_name in image_train_list:
            f.write(image_name + "\n")
        image_test_list.sort()
        f.write("\ntest_list:\n")
        for image_name in image_test_list:
            f.write(image_name + "\n")

    for epoch in range(10):
        # train model
        random.shuffle(image_train_list)
        for index, image_name in enumerate(image_train_list, 1):
            # load image
            image = h5py.File(os.path.join(dir, "hyperspectral", image_name))
            image_data = np.expand_dims(image["hypercube"][()].astype('float32'), axis=0)  # shape: [1, 81, 1024, 768]
            data = image_data / np.max(image_data)

            data = torch.from_numpy(data)
            data = Variable(data)

            # model


            if not model_flag:
                #model = rf.RFNet(data.size(1), nChannels, 1)
                model=net.TestNet()
                model.train()
                loss_fn = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.8)
                model_flag = True

            # forward
            optimizer.zero_grad()
            output = model(data)[0]  # shape: [args.nChannels, 1024, 768]
            output = output.permute(2, 1, 0).contiguous().view(-1, 2)  # shape: [768*1024, args.nChannels]

            # output = output.permute(2, 1, 0)
            # output = output.permute(2, 1, 0)
            # img=output.detach().numpy()
            # cv2.imshow('image',img[:,:,0])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.imshow('image', img[:,:,1])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # output=output.contiguous().view(-1,1)

            # _, target = torch.max(output, 1)
            # target = target.data.cpu().numpy()
            # nLabels = len(np.unique(target))

            # vote
            # salincy_map=[]
            # threshold=1
            # for i in range(768):
            #     for j in range(1024):
            #         s=0
            #         for k in range(2):
            #             s=output[i][j][k]+s
            #         if s>threshold:
            #             salincy_map.append(1)
            #         else:
            #             salincy_map.append(0)

            image_num=image_name.split('.')
            img=image_num[0]+'.jpg'
            ground_truth_path = os.path.join(dir, 'ground_truth',img)
            ground_truch=cv2.imread(ground_truth_path,cv2.IMREAD_GRAYSCALE)
            ground_truch = cv2.threshold(ground_truch, 0.49, 1, cv2.THRESH_BINARY)
            ground_truch=ground_truch[1]
            # ground_truch=np.array(ground_truch)
            # ground_truch=torch.tensor(ground_truch)
            ground_truch=torch.from_numpy(ground_truch)

            ground_truch=ground_truch.contiguous().view(1,-1).float()

            print(ground_truch[torch.argmax(ground_truch)])
            cv2.imshow('image',ground_truch)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # loss and backward
            # target = torch.from_numpy(target)
            # target = Variable(target)
            # loss = loss_fn(output, target)
            ground_truch=Variable(ground_truch)
            loss=loss_fn(output,ground_truch[0].long())
            print(loss)
            loss.backward()
            optimizer.step()



        model_name = "MBCNN" + "_epoch_%d.pkl" % (epoch + 1)
        torch.save(model.state_dict(), os.path.join('./net/', model_name))







