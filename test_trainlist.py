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

net_num=5
test_net_No=1
evaluation_file='evaluation.txt'
for test_net_No in range(20):
    test_net_No += 21
    branch_num = 1
    branch_No=np.random.randint(low=1,high=71,size=net_num,dtype='int')
    branch_layer = 5
    branch_net = 'End_to_End_L' + str(branch_layer)
    hyperspectral_channels = 81
    vote_rule = 0
    sample_channels = 54
    loss_function = ''

    base_net_name = branch_net + '_BN' + str(branch_num)
    information_file = 'information.txt'
    dataset_dir = './dataset/HS-SOD/'
    base_output_path = './output/'
    first_net_path = './net/'
    channel_file = "channel.txt"

    dir_txt = os.path.join(first_net_path, information_file)
    flag=True
    with open(dir_txt, "r") as f:
        while(flag):
            text=f.readline()
            if(text=="train_list:\n"):
                image_test_list_str=f.read()
                flag=False

    image_test_list=image_test_list_str.split("\n\n")
    image_test_list=image_test_list[0].split('\n')
    image_test_list.pop()

    use_cuda=torch.cuda.is_available()
    sum_test_nLabels = 0
    sum_test_loss = 0
    image_test_num=len(image_test_list)
    loss_fn = torch.nn.CrossEntropyLoss()

    model=[]
    channel_path=[]
    channels = net.Sample(net_num)

    base_net_file=base_net_name+'_TestNo.'+str(test_net_No)
    first_output_path=os.path.join(base_output_path,base_net_file+'/')
    net_info_file='info.txt'
    net_info_path=os.path.join(first_output_path,'net_info/')              #网络信息
    net_info_dir=os.path.join(net_info_path,net_info_file)
    net_result_path=os.path.join(first_output_path,'result/')
    net_train_result_path=os.path.join(net_result_path,'result_train_list/')
    evaluation_dir=os.path.join(net_info_path,evaluation_file)
    if not os.path.exists(first_output_path):
        os.mkdir(first_output_path)
        os.mkdir(net_info_path)
        os.mkdir(net_result_path)
        os.mkdir(net_train_result_path)
        with open(net_info_dir,'w+') as f:
            f.write("channel,model_dir\n")
        with open(evaluation_dir,'w+') as f:
            pass
    # =================================================================
    for i in range(net_num):
        No = branch_No[i]
        # net_name=f'{branch_net}_B{str(branch_num)}'
        net_name = base_net_name + '_No.' + str(No)
        second_net_path = os.path.join(first_net_path, base_net_name + '/')
        net_path = os.path.join(second_net_path, net_name + '/')
        net_dir=os.path.join(net_path, channel_file)
        channel_path.append(net_dir)
        # =========================================
        model_file=net_name + '_B1_epoch10.pkl'
        model_path=os.path.join(net_path, model_file)
        # =========================================
        strs=[]
        if not os.path.exists(net_dir):
            print(net_dir)
        with open(net_dir,"r+") as f:
            strs=f.readline()
        with open(net_info_dir, "a") as f:
            strs=strs[0:-1].replace(',',':')
            strs=strs.replace(' ','')
            f.write(strs+',')
            f.write(model_path + '\n')

        # 加载模型
        model.append(net.TestNet())
        model[i].load_state_dict(torch.load(model_path))
        channels.calc_sample2(channel_path[i])
    # =================================================================

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
        output=[]
        for i in range(net_num):
            ch = channels.get_sample()
            sample=data[:,ch[i]]
            sample = torch.from_numpy(sample)
            sample = Variable(sample)
            map=model[i](sample)
            map=map.permute(0,1,3,2)
            output.append(map)  # shape: [args.nChannels, 1024, 768]

        vote=net.Vote(output)
        saliency_map=vote.vote(0)

        image_num = image_name.split('.')
        img = image_num[0] + '.jpg'
        saliancy_path=os.path.join(net_train_result_path,img)
        cv2.imwrite(saliancy_path, saliency_map[1]*255)

        # ground_truth
        image_num = image_name.split('.')
        img = image_num[0] + '.jpg'
        ground_truth_path = os.path.join(dataset_dir, 'ground_truth', img)
        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        ground_truth = cv2.threshold(ground_truth, 0.49, 1, cv2.THRESH_BINARY)
        ground_truth = ground_truth[1]
        ground_truth=ground_truth.reshape(-1,1)
        # ground_truth = torch.from_numpy(ground_truth)
        # ground_truth = ground_truth.contiguous().view(1, -1).float()

        #evaluate
        saliency_map=saliency_map[1].reshape(-1,1)
        target_name=['class0','class1']
        report=classification_report(ground_truth,saliency_map,target_names=target_name)
        auc=roc_auc_score(ground_truth,saliency_map)
        mae=mean_absolute_error(ground_truth,saliency_map)

        with open(evaluation_dir,'a') as f:
            f.write(image_name+'\n')
            f.write("========================================\n")
            f.write("report:\n"+report+'\n')
            f.write("AUC: "+str(auc)+'\n')
            f.write("MAE: "+str(mae)+'\n\n')

    print("[Test] Avg nLabels: %.2f, Avg Loss: %.4f\n" %
          (sum_test_nLabels / image_test_num, sum_test_loss / image_test_num))