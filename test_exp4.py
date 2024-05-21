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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from pathlib import Path,PurePath
import Datasets
import Process

def net_name(head,No):
    return head+str(No)

class Netlib():
    def __init__(self):
        self.net_type=""
        self.net_No=""
        self.net_epoch=0
        self.net_branch=0
        self.net_name=""

    def set_Netlib(self,type="default",No=1,epoch=1,branch=1):
        self.net_type = type
        self.net_No = No
        self.net_epoch = epoch
        self.net_branch = branch

    def set_type(self,type):
        self.net_type = type

    def set_No(self,No):
        self.net_No=No

    def set_epoch(self,epoch):
        self.net_epoch=epoch

    def set_branch(self,branch):
        self.net_branch=branch

    def set_name(self,name):
        self.net_name=name

    def get_Nettype(self):
        return self.net_type

    def get_NetNo(self):
        return self.net_type+"_No."+str(self.net_No)

    def get_Netbranch(self):
        return self.net_type+str(self.net_No)+"_B"+str(self.net_branch)

    def get_Netepoch(self):
        return self.net_type + str(self.net_No) + "_B"+str(self.net_branch)+"_epoch"+str(self.net_epoch)

    def get_Netpkl(self):
        return self.net_type + str(self.net_No) + "_B"+str(self.net_branch)+"_epoch"+str(self.net_epoch)+".pkl"



def search(location,file):
    path=Path(location)
    path_list=[]
    for item in path.rglob(file):
        path_list.append(item)
    return path_list

def select_net_random(net_list,select_num):
    selected_net=[]
    length=len(net_list)
    for item in range(select_num):
        index=random.randint(0,length-1)
        selected_net.append(net_list[index])

    return selected_net

def get_pkllist(location=PurePath(Path.cwd(),"net")):

    path=Path(location)
    path=path.rglob(location,"*epoch25.pkl")
    pkl_list=[]
    for dir in path:
        pkl_list.append(dir)

    return pkl_list

def get_netpath(location):
    netpath=location.parent
    return PurePath(netpath)

def get_channeltxt(location):
    file=PurePath(location,"channel.txt")
    strs=""
    with open(file,mode="r") as f:
        strs=f.read()
    channel_list=strs.split("]")
    channel_list=channel_list[0].split("[")
    channel_list = channel_list[-1].split(",")
    return channel_list

net_num=4
test_net_No=1
cwd=Path.cwd()
evaluation_file='evaluation.txt'
channel_file = "channel.txt"
dataset_dir = PurePath(cwd,'dataset\\HS-SOD\\')
hyperspectral_path=PurePath(dataset_dir,"hyperspectral")
gt_path=PurePath(dataset_dir,"ground_truth")
net_path=PurePath(cwd,"net")
#修改目录
#=======================================================================
net_path=PurePath(net_path,"temp")
#=======================================================================
output_path=PurePath(cwd,"output")
sample_channels = 54
hyperspectral_channels = 81

netlib=Netlib()

test_net_No=1
test_net_name="End_to_End_L5_BN1"
test_net_each_path=test_net_name+str(test_net_No)

model_file="*epoch25.pkl"
netpaths=Path(net_path)
branch_net_list=[]
for bnet in netpaths.rglob(model_file):
    branch_net_list.append(bnet)
testlib=Process.Testlib(net_num)



#test_list:
information_file=search(branch_net_list[0].parent,"information.txt")
for file in information_file:
    with open(file,"r") as f:
        img_list=f.read()
    break
img_process=Process.process()
img_process.read_imageList(img_list)
img_list=img_process.image_test_list

#ground_truth path
gt_paths=Path(gt_path)
gt_paths=gt_paths.glob("*.jpg")
gt_dirs=[]
for item in gt_paths:
    gt_dirs.append(item)

print(branch_net_list)
print(gt_dirs)
for test_net_No in range(10):
    test_net_No +=31
    #确定分支数量
    #确定预训练模型r
    #确定通道
    sub_path = test_net_name + "_No." + str(test_net_No)
    selected_net = select_net_random(branch_net_list, net_num)
    print("==================================================")
    print(sub_path + " selected the follow branch:")
    for item in selected_net:
        print(item,end="")
    print("\n==================================================")
    for item in selected_net:
        parent_path = item.parent
        channel = get_channeltxt(parent_path)
        testlib.add_channel([int(item) for item in channel])

    # branch_nets=[]
    # for branch in range(net_num):
    #     #加载分支模型
    #     branch_nets.append(net.TestNet())
    #     torch.load(selected_net[branch])

    #model
    model = net.MBCNN(branch_num=net_num, channels=testlib.channels)
    model.set_parameters(lr=0.001, momentum=0.90)
    for i in range(net_num):
        #加载分支模型
        model.model[i].load_state_dict(torch.load(selected_net[i]))

    for img_dir in img_list:
        img_path=PurePath(hyperspectral_path,img_dir)
        img_data=h5py.File(img_path)
        #预处理
        img_data=img_process.img_process(img_data)
        #run model
        output=model.myforward(img_data)
        for i in range(len(output)):
            output[i]=output[i].cpu()
        vote=net.Vote(output)
        # saliency_maps.append(vote.vote(0))
        saliency_map=vote.probability_map()

        img_name= PurePath(img_dir).stem+".jpg"
        save_dir=Path(output_path,sub_path)
        save_path=PurePath(save_dir,img_name)
        save_dir.mkdir(exist_ok=True)               #exist_ok=True:文件已存在，忽略异常
        cv2.imwrite(str(save_path),saliency_map)
        print("img:{}".format(img_dir)+" has been saved at {}".format(str(save_path)))







