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
import FileManager as fm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error

def trainning(fileManager,name,epochs):
    f = fileManager
    for epoch in range(epochs):
        # train model
        epoch_flag=True
        random.shuffle(f.image_train_list)
        for index, image_name in enumerate(f.image_train_list, 1):
            # load image
            f.setcwd(f.paths["dataset_hyperspectral_path"])  # ++++++++++++++++++++++++++
            image = f.readFile(image_name, mode='r')
            image_data = np.expand_dims(image["hypercube"][()].astype('float32'), axis=0)  # shape: [1, 81, 1024, 768]
            data = image_data / np.max(image_data)
            # model
            if not name.model_flag:
                model = []
                loss_fn = []
                optimizer = []
                for i in range(name.branch_num):
                    model.append(net.TestNet())
                    model[i].train()
                    loss_fn.append(torch.nn.CrossEntropyLoss())
                    optimizer.append(torch.optim.SGD(model[i].parameters(), lr=0.001, momentum=0.90))
                name.model_flag = True
            # forward
            output = []
            # sample的改变是否会影响链式求导？
            for i in range(name.branch_num):
                optimizer[i].zero_grad()
                sample = data[:, name.channel_list[i]]
                sample = torch.from_numpy(sample)
                sample = Variable(sample)
                x = model[i](sample)  # shape: [args.nChannels, 1024, 768]
                x=x.permute(0,1,3,2)
                output.append(x.permute(0, 2, 3, 1).contiguous().view(1, -1, 2))
            # ground_truth
            image_num = image_name.split('.')
            img = image_num[0] + '.jpg'
            f.setcwd(f.paths["dataset_ground_truth_path"])
            ground_truth = f.readFile(img, mode=cv2.IMREAD_GRAYSCALE)
            ground_truth = cv2.threshold(ground_truth, 0.49, 1, cv2.THRESH_BINARY)
            ground_truth = ground_truth[1]
            ground_truth = torch.from_numpy(ground_truth)
            ground_truth = ground_truth.contiguous().view(1, -1).float()

            # ground_truth需要求梯度？
            # loss 只能用一次
            ground_truth = Variable(ground_truth)
            print(f'epoch{epoch + 1}:', end='')
            for i in range(name.branch_num):
                loss = loss_fn[i](output[i][0], ground_truth[0].long())
                print('loss{}={};   '.format(str(i + 1), str(loss)), end='')
                loss.backward()
                optimizer[i].step()
            print(end='\n')

            # 保存训练时显著图与评价指标
            # evaluate
            saliency = torch.max(x.data, dim=1)
            saliency = saliency[1].contiguous().view(1, -1).permute(1, 0)
            saliency = saliency.detach().numpy()
            ground_truth=ground_truth.permute(1,0)
            ground_truth=ground_truth.detach().numpy()
            target_name = ['class0', 'class1']
            report = classification_report(ground_truth, saliency, target_names=target_name)
            auc = roc_auc_score(ground_truth, saliency)
            mae = mean_absolute_error(ground_truth, saliency)
            sign="========================================\n"
            strs=image_name+"\n"+sign+"report:\n"+report+"\n"+"AUC: "+str(auc)+ '\n'+"MAE: " + str(mae) + '\n\n'
            f.setcwd(name.second_net_path)
            if epoch==epochs:
                f.writeFile(name.evaluationFile,strs,mode='a')
            f.setcwd("result")
            if epoch_flag:
                f.create_path(f"epoch{epoch}")
                f.setcwd(f"epoch{epoch}")
                f.create_file(name.evaluationFile)
                epoch_flag=False
            else:
                f.setcwd(f"epoch{epoch}")
            f.writeFile(name.evaluationFile,strs,mode='a')
            im_name = image_name.split(".")
            saliency=torch.max(x.data, dim=1)
            saliency=saliency[1]
            saliency=saliency.detach().numpy()
            f.writeFile(im_name[0] + ".jpg", saliency[0] * 255, mode="w")

        # save net
        # model_name = net_name + "_epoch_%d.pkl" % (epoch + 1)
        for i in range(name.branch_num):
            model_name = name.get_netName() + "_B{}_epoch{}.pkl".format(str(i + 1), str(epoch + 1))
            f.setcwd(f.paths["second_net_path"])
            torch.save(model[i].state_dict(), os.path.join(f.current_path, model_name))
            print('save model {}'.format(model_name))


def initial_File(name,fileManager):
    f=fileManager
    #路径
    f.setcwd()
    f.create_path(name.output_path)
    f.create_path(name.net_path)
    #创建训练集与测试集文件
    f.create_file(name.set_listFile)                #有bug
    #网络名称
    train_netName = name.get_name()
    #一级路径
    f.setcwd(name.net_path)
    f.create_path(train_netName)
    name.first_net_path = f.current_file
    # 分配训练集与测试集
    #====================================================
    f.setcwd(name.dataset_hyperspectral_path)
    image_list = f.current_file_list
    f.create_SetList(image_list, name.first_net_path)
    # ====================================================
    f.setcwd(name.first_net_path)
    name.set_No(name.No)
    name.second_net_path = name.get_netName()
    f.create_path(name.second_net_path)
    f.setcwd(name.second_net_path)
    name.second_net_path = f.current_path
    f.create_SetList(image_list, name.second_net_path)
    f.add_path([name.dataset_hyperspectral_path, name.dataset_ground_truth_path, name.second_net_path], \
               ["dataset_hyperspectral_path", "dataset_ground_truth_path", "second_net_path"])
    # channels
    # ==========================================================
    f.create_file(name.channelFile)
    data = []
    lists = list(range(0, 81))
    for i in range(name.branch_num):
        ch_list = random.sample(lists, name.sample_channels)
        ch_list.sort()
        data.append(str(ch_list))
        name.channel_list.append(ch_list)
    f.writeFile(name.channelFile, "\n".join(data), mode="w+")
    f.create_file(name.evaluationFile)
    f.create_path("result")
    name.model_flag=False


if __name__=="__main__":

    name = fm.Names(net="End_to_End", trainning=True)
    name.set_Name(net="End_to_End",layer=5,branch_num=1,No=1,trainning=True)
    f=fm.File()
    for i in range(5):
        name.set_No(i+6)
        initial_File(name,f)
        trainning(f,name,25)           #添加显著图和评价指标












# net_path='net'
# output_path='output'
#
# first_net_path=""
# first_output_path=""
#
# second_net_path=""
# second_output_path=""
#
# saliency_output_path=""
# resultInfo_output_path=""
#
# dataset_path='dataset'
# dataset_HSSOD_path='HS-SOD'
# # dataset_hyperspectral_path='hyperspectral'
# dataset_hyperspectral_path=".\\dataset\\HS-SOD\\hyperspectral\\"
# dataset_ground_truth_path=".\\dataset\\HS-SOD\\ground_truth\\"
#
# evaluationFile='evaluation.txt'
# netInfoFile='info.txt'
# channelFile='channel.txt'
# set_listFile='information.txt'
# No=1
# branch_num=3
# sample_channels=54
# model_flag = False


