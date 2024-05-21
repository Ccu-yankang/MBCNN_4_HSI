import torch
torch.cuda.current_device()         #RuntimeError: CUDA error: unknown error
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import cv2
from torch.autograd import Variable
import Evaluate as eva
from torchmetrics import Accuracy,F1Score,Recall,Precision,Specificity
import math

import net                      #有用？

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet,self).__init__()

        self.conv1 = nn.Conv2d(54, 34, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(54)

        self.conv2 = nn.Conv2d(34, 15, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(34)

        self.conv3 = nn.Conv2d(15, 2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(15)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.ct1 = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(2)

        self.ct2 = nn.ConvTranspose2d(2, 2, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        # x=F.max_pool2d(self.bn1(F.relu(self.conv1(x),inplace=True)),kernel_size=2,stride=2)
        # x=F.max_pool2d(self.bn2(F.relu(self.conv2(x),inplace=True)),kernel_size=2,stride=2)
        # x=self.bn3(F.relu(self.conv3(x),inplace=True))
        # x=self.bn3(F.relu(self.ct1(self.up(x)),inplace=True))
        # x=self.bn3(F.relu(self.ct2(self.up(x)),inplace=True))

        #============================================================================================
        # x = torch.from_numpy(x)
        # x = Variable(x)
        x = F.max_pool2d(F.relu(self.conv1(self.bn1(x)), inplace=True), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(self.bn2(x)), inplace=True), kernel_size=2, stride=2)
        x = F.relu(self.conv3(self.bn3(x)), inplace=True)
        x = F.relu(self.ct1(self.up(self.bn4(x))), inplace=True)
        x = F.relu(self.ct2(self.up(self.bn4(x))))
        x = x.softmax(dim=1)
        return x

class BranchNet_L5_with_MSE(nn.Module):
    def __init__(self):
        super(BranchNet_L5_with_MSE, self).__init__()
        self.conv1 = nn.Conv2d(54, 34, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(54)

        self.conv2 = nn.Conv2d(34, 15, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(34)

        self.conv3 = nn.Conv2d(15, 1, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(15)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.ct1 = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(2)

        self.ct2 = nn.ConvTranspose2d(2, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(self.bn1(x)), inplace=True), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(self.bn2(x)), inplace=True), kernel_size=2, stride=2)
        x = F.relu(self.conv3(self.bn3(x)), inplace=True)
        x = F.relu(self.ct1(self.up(self.bn4(x))), inplace=True)
        x = F.relu(self.ct2(self.up(self.bn4(x))))
        # x = x.softmax(dim=1)
        return x



class TestNet_L7(nn.Module):
    def __init__(self):
        super(TestNet_L7,self).__init__()

        self.conv1 = nn.Conv2d(54, 40, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(54)

        self.conv2 = nn.Conv2d(40, 25, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(40)

        self.conv3 = nn.Conv2d(25, 10, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(25)

        self.conv4 = nn.Conv2d(10, 2, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(10)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.ct1 = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(2)

        self.ct2 = nn.ConvTranspose2d(2, 2, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        # x=F.max_pool2d(self.bn1(F.relu(self.conv1(x),inplace=True)),kernel_size=2,stride=2)
        # x=F.max_pool2d(self.bn2(F.relu(self.conv2(x),inplace=True)),kernel_size=2,stride=2)
        # x=self.bn3(F.relu(self.conv3(x),inplace=True))
        # x=self.bn3(F.relu(self.ct1(self.up(x)),inplace=True))
        # x=self.bn3(F.relu(self.ct2(self.up(x)),inplace=True))

        #============================================================================================
        # x = torch.from_numpy(x)
        # x = Variable(x)
        x = F.max_pool2d(F.relu(self.conv1(self.bn1(x)), inplace=True), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(self.bn2(x)), inplace=True), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv3(self.bn3(x)), inplace=True), kernel_size=2, stride=2)
        x = F.relu(self.conv4(self.bn4(x)), inplace=True)
        x = F.relu(self.ct1(self.up(self.bn5(x))), inplace=True)
        x = F.relu(self.ct1(self.up(self.bn5(x))), inplace=True)
        x = F.relu(self.ct2(self.up(self.bn5(x))))
        x = x.softmax(dim=1)
        return x

class BranchNet_L7_with_MSE(nn.Module):
    def __init__(self):
        super(BranchNet_L7_with_MSE,self).__init__()

        self.conv1 = nn.Conv2d(54, 40, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(54)

        self.conv2 = nn.Conv2d(40, 25, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(40)

        self.conv3 = nn.Conv2d(25, 10, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(25)

        self.conv4 = nn.Conv2d(10, 1, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(10)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.ct1 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(1)

        self.ct2 = nn.ConvTranspose2d(1, 1, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(self.bn1(x)), inplace=True), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(self.bn2(x)), inplace=True), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv3(self.bn3(x)), inplace=True), kernel_size=2, stride=2)
        x = F.relu(self.conv4(self.bn4(x)), inplace=True)
        x = F.relu(self.ct1(self.up(self.bn5(x))), inplace=True)
        x = F.relu(self.ct1(self.up(self.bn5(x))), inplace=True)
        x = F.relu(self.ct2(self.up(self.bn5(x))))
        #一个通道下不能用softmax,用了就全为1了
        # x = x.softmax(dim=1)
        return x

class MBCNN(nn.Module):
    def __init__(self,branch_num,channels,use_MSE=False):
        self.use_MSE=use_MSE
        self.model=[]
        self.loss_fn=[]
        self.loss_crossentropy=[]
        self.loss_MSE=[]
        self.loss_precision=[]
        self.loss_recall=[]
        self.optimizer=[]
        self.lr=0.001
        self.momentum=0.90
        self.branch_num=branch_num
        self.channels=channels
        self.use_cuda=torch.cuda.is_available()
        # self.use_cuda = False
        for i in range(branch_num):
            self.model.append(net.BranchNet_L7_with_MSE())
            self.loss_precision.append(Precision(average='weighted', num_classes=2))
            self.loss_recall.append(Recall(average='weighted', num_classes=2))
            if self.use_cuda:
                self.model[i].cuda()
                self.loss_precision[i].cuda()
                self.loss_recall[i].cuda()
            if use_MSE:
                self.loss_MSE.append(torch.nn.MSELoss())
            else:
                self.loss_crossentropy.append(torch.nn.CrossEntropyLoss())

            self.optimizer.append(torch.optim.SGD(self.model[i].parameters(), lr=self.lr, momentum=self.momentum))
        self.precision=0
        self.recall=0
        self.loss_nn_info=""
        self.loss_crossentropy_info="NA"
        self.loss_MSE_info="NA"
        self.acos_loss_crossentropy_info=""
        self.acos_loss_MSE_info = ""
        self.loss_precision_info=""
        self.acos_loss_precision_info = ""
        self.loss_recall_info=""
        self.acos_loss_recall_info = ""
        self.loss_fn_info=""
        self.loss_info=""

    def set_parameters(self,lr,momentum):
        self.lr=lr
        self.momentum=momentum

    def myforward(self,data):
        output=[]
        for i in range(self.branch_num):
            self.optimizer[i].zero_grad()
            sample=data[:,self.channels[i]]
            sample=torch.from_numpy(sample)
            if self.use_cuda:
                sample=sample.cuda()
            sample=Variable(sample)
            output.append(self.model[i](sample))
            # x=self.model[i](sample)
            # output.append(x.permute(0,2,3,1).contiguous().view(1,-1,2))
        return output

    def mybackward(self,output,ground_truth):
        ground_truth=torch.from_numpy(ground_truth)
        if self.use_cuda:
            ground_truth=ground_truth.cuda()
        ground_truth = Variable(ground_truth).float()
        ground_truth_2D=Variable(ground_truth).float()
        ground_truth = ground_truth.contiguous().view(1, -1)
        ground_truth = ground_truth[0]
        for i in range(self.branch_num):
            #损失函数
            if self.use_MSE:
                loss_nn = self.loss_MSE[i](output[i][0][0], ground_truth_2D)
                output[i]=output[i].permute(0,2,3,1).contiguous().view(1,-1)
                # output[i]=torch.where(output[i] >0.5,1,0).long()       #失去梯度属性
            else:
                output[i] = output[i].permute(0, 2, 3, 1).contiguous().view(1, -1, 2)
                loss_nn = self.loss_crossentropy[i](output[i][0], ground_truth)

            if not self.use_MSE:
                loss_precision=self.loss_precision[i](output[i][0],ground_truth)
                #average="binary"(即“none”)时，需要loss_precisino[1]
                # loss_precision=loss_precision[1]
                acos_loss_precision = 2*torch.acos(loss_precision)/math.pi
                loss_recall=self.loss_recall[i](output[i][0],ground_truth)
                # average="binary"(即“none”)时，需要loss_recall[1]
                # loss_recall=loss_recall[1]
                acos_loss_recall = 2*torch.acos(loss_recall)/math.pi
                loss_fn=0.1*loss_nn+0.45*acos_loss_precision+0.45*acos_loss_recall
            else:
                loss_fn=loss_nn
            if self.use_MSE:
                self.loss_MSE_info = 'branch{}:loss_MSE           ={}'.format(str(i + 1), \
                loss_nn)
            else:
                self.loss_crossentropy_info='branch{}:loss_crossentropy  ={}'.format(str(i + 1), \
                loss_nn)

            self.loss_nn_info=              'branch{}:{}={}'.format(str(i + 1), \
                (lambda loss_name:"loss_MSE           " if loss_name else "loss_crossentropy  ")(self.use_MSE),\
                loss_nn)


            if not self.use_MSE:
                self.loss_precision_info =      'branch{}:loss_precision     ={}'.format(str(i + 1), str(loss_precision))
                self.acos_loss_precision_info = 'branch{}:acos_loss_precision={}'.format(str(i + 1), str(acos_loss_precision))
                self.loss_recall_info =         'branch{}:loss_recall        ={}'.format(str(i + 1), str(loss_recall))
                self.acos_loss_recall_info =    'branch{}:acos_loss_recall   ={}'.format(str(i + 1), str(acos_loss_recall))

                self.loss_info=self.loss_crossentropy_info+"\n"+self.loss_MSE_info+"\n"+self.loss_precision_info+"\n"+self.acos_loss_precision_info+"\n"+self.loss_recall_info+"\n"+self.acos_loss_recall_info+"\n"+self.loss_fn_info

                print(self.loss_precision_info, end=';  ')
                print(self.loss_recall_info, end=';')

            self.loss_fn_info = 'branch{}:loss_fn            ={}'.format(str(i + 1), str(loss_fn))
            print(self.loss_nn_info, end=';  ')
            loss_fn.backward()
            self.optimizer[i].step()
        print(end="\n")

    def run(self,data,ground_truth):
        for i in range(self.branch_num):
            self.optimizer[i].zero_grad()
        x=self.myforward(data)
        saliency=[]
        if not self.use_MSE:
            saliency=torch.max(x[0],dim=1)
        else:
            for i in range(len(x)):
                saliency.append(x[i].clone().detach())
        self.mybackward(x,ground_truth)
        saliency_map=[]
        if self.use_cuda:
            for i in range(len(saliency)):
                saliency_map.append(saliency[i].cpu())
        else:
            for i in range(len(saliency)):
                saliency_map.append(saliency[i])
        # precision=Precision(average="macro",num_classes=2)
        # ground_truth = torch.from_numpy(ground_truth)
        # if self.use_cuda:
        #     ground_truth = ground_truth.cuda()
        #     precision.cuda()
        # ground_truth = Variable(ground_truth)
        # ground_truth=ground_truth.contiguous().view(1,-1)
        # saliency=saliency[1].contiguous().view(1, -1).permute(1, 0)
        #
        # self.precision=precision(saliency,ground_truth)
        # print("precision:"+str(self.precision))
        return saliency_map

sample_path='./net/channel.txt'
class Sample():
    def __init__(self,branch_num,path=sample_path):
        self.path=path
        self.branch_num=branch_num
        self.branch=[]                      #分支样本

    def calc_sample(self):
        with open(self.path, 'r+') as f:
            for i in range(self.branch_num):
                ch=f.readline()
                ch=ch[1:-2]
                ch=ch.split(',')
                ch=list(map(int,ch))
                self.branch.append(ch)

    def calc_sample2(self,channel_path):
        with open(channel_path,'r+') as f:
            ch=f.readline()
            ch=ch[1:-2]
            ch = ch.split(',')
            ch = list(map(int, ch))
            self.branch.append(ch)


    def get_sample(self):
        return self.branch

class Vote():
    def __init__(self,data):
        self.data=data
        self.branch_num=len(data)

    def vote(self,method=0):
        if method==0:
            saliency_map=np.zeros([768,1024])
            for i in range(self.branch_num):
                output = torch.max(self.data[i], dim=1)
                # output = 1 - output[1]
                # output = output[1].permute(1, 0)
                output = output[1]
                saliency_map=output+saliency_map
            saliency_map=saliency_map.detach().numpy()
            saliency_map=cv2.threshold(saliency_map[0], int(self.branch_num/2), 1, cv2.THRESH_BINARY)
            return saliency_map

    def probability_map(self):
        p_map=np.zeros([768,1024])
        probability_map=np.zeros([768,1024])
        for i in range(self.branch_num):
            output=torch.max(self.data[i],dim=1)
            output=output[1]
            output=output[0]
            p_map=output+p_map
        p_map=p_map.detach().numpy()
        cv2.normalize(p_map,probability_map,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)
        return probability_map

class branch(TestNet):
    def __init__(self,channels,model_path):
        self.channels=channels
        self.net = TestNet()
        self.net=self.net.load_state_dict(torch.load(model_path))
        self.map=[]

    def run(self,data, ground_truth):
        self.map=self.net(data)
        return self.map



class RFNet(nn.Module):
    def __init__(self,input_dim,nChannels,branch_number):
        super(RFNet,self).__init__()
        self.branch_number=branch_number
        self.branch=[]
        self.tree=nn.Sequential(
            nn.Conv2d(input_dim, nChannels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(nChannels),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(nChannels, nChannels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(nChannels),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(nChannels, nChannels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(nChannels),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(nChannels, nChannels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(nChannels),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(nChannels, nChannels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(nChannels)

        )

    def forward(self,x):
        for i in range(self.branch_number):
            self.branch.append(self.tree(x))
        return self.branch(0)




class TestNet2(nn.Module):
    def __init__(self):
        super(TestNet2, self).__init__()

        self.conv1 = nn.Conv2d(81, 50, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(81)

        self.conv2 = nn.Conv2d(50, 20, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(50)

        self.conv3 = nn.Conv2d(20, 2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(20)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.ct1 = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(2)

        self.ct2 = nn.ConvTranspose2d(2, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x, channel):
        # branch1
        x1 = F.max_pool2d(F.relu(self.conv1(self.bn1(x[channel[0]])), inplace=True), kernel_size=2, stride=2)
        x1 = F.max_pool2d(F.relu(self.conv2(self.bn2(x1)), inplace=True), kernel_size=2, stride=2)
        x1 = F.relu(self.conv3(self.bn3(x1)), inplace=True)
        x1 = F.relu(self.ct1(self.up(self.bn4(x1))), inplace=True)
        x1 = F.relu(self.ct2(self.up(self.bn4(x1))))
        x1 = x1.softmax(dim=1)

        # x1 = F.max_pool2d(self.bn1(self.conv1(self.bn1(x[channel[0]])),inplace=True),kernel_size=2,stride=2)
        # x1 = F.max_pool2d(F.relu(self.conv2(self.bn2(x1)),inplace=True),kernel_size=2,stride=2)
        # x1 = F.relu(self.conv3(self.bn3(x1)),inplace=True)
        # x1 = F.relu(self.ct1(self.up(self.bn3(x1))),inplace=True)
        # x1 = F.relu(self.ct2(self.up(self.bn3(x1))),inplace=True)

        # branch2
        x2 = F.max_pool2d(F.relu(self.conv1(self.bn1(x[channel[1]])), inplace=True), kernel_size=2, stride=2)
        x2 = F.max_pool2d(F.relu(self.conv2(self.bn2(x2)), inplace=True), kernel_size=2, stride=2)
        x2 = F.relu(self.conv3(self.bn3(x2)), inplace=True)
        x2 = F.relu(self.ct1(self.up(self.bn4(x2))), inplace=True)
        x2 = F.relu(self.ct2(self.up(self.bn4(x2))))
        x2 = x2.softmax(dim=1)

        # x2 = F.max_pool2d(self.bn1(self.conv1(self.bn1(x[channel[1]])), inplace=True), kernel_size=2, stride=2)
        # x2 = F.max_pool2d(F.relu(self.conv2(self.bn2(x2)), inplace=True), kernel_size=2, stride=2)
        # x2 = F.relu(self.conv3(self.bn3(x2)), inplace=True)
        # x2 = F.relu(self.ct1(self.up(self.bn3(x2))), inplace=True)
        # x2 = F.relu(self.ct2(self.up(self.bn3(x2))), inplace=True)

        return [x1, x2]

class TestNet3(nn.Module):
    def __init__(self):
        super(TestNet3,self).__init__()

        self.conv1 = nn.Conv2d(54, 34, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(54)

        self.conv2 = nn.Conv2d(34, 15, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(34)

        self.conv3 = nn.Conv2d(15, 2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(15)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.ct1 = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(2)

        self.ct2 = nn.ConvTranspose2d(2, 2, kernel_size=1, stride=1, padding=0)


    def forward(self,x,channel):
        #branch1
        x1=x[:,channel[0]]
        x1 = torch.from_numpy(x1)
        x1 = Variable(x1)
        x1 = F.max_pool2d(F.relu(self.conv1(self.bn1(x1)), inplace=True), kernel_size=2, stride=2)
        x1 = F.max_pool2d(F.relu(self.conv2(self.bn2(x1)), inplace=True), kernel_size=2, stride=2)
        x1 = F.relu(self.conv3(self.bn3(x1)), inplace=True)
        x1 = F.relu(self.ct1(self.up(self.bn4(x1))), inplace=True)
        x1 = F.relu(self.ct2(self.up(self.bn4(x1))))
        x1 = x1.softmax(dim=1)

        #branch2
        x2=x[:,channel[1]]
        x2 = torch.from_numpy(x2)
        x2 = Variable(x2)
        x2 = F.max_pool2d(F.relu(self.conv1(self.bn1(x2)), inplace=True), kernel_size=2, stride=2)
        x2 = F.max_pool2d(F.relu(self.conv2(self.bn2(x2)), inplace=True), kernel_size=2, stride=2)
        x2 = F.relu(self.conv3(self.bn3(x2)), inplace=True)
        x2 = F.relu(self.ct1(self.up(self.bn4(x2))), inplace=True)
        x2 = F.relu(self.ct2(self.up(self.bn4(x2))))
        x2 = x2.softmax(dim=1)

        # branch3
        x3 = x[:, channel[2]]
        x3 = torch.from_numpy(x3)
        x3 = Variable(x3)
        x3 = F.max_pool2d(F.relu(self.conv1(self.bn1(x3)), inplace=True), kernel_size=2, stride=2)
        x3 = F.max_pool2d(F.relu(self.conv2(self.bn2(x3)), inplace=True), kernel_size=2, stride=2)
        x3 = F.relu(self.conv3(self.bn3(x3)), inplace=True)
        x3 = F.relu(self.ct1(self.up(self.bn4(x3))), inplace=True)
        x3 = F.relu(self.ct2(self.up(self.bn4(x3))))
        x3 = x3.softmax(dim=1)

        output=torch.cat((x1,x2,x3),dim=0)
        return output

class TestNet5(nn.Module):
    def __init__(self):
        super(TestNet5,self).__init__()

        self.conv1 = nn.Conv2d(54, 34, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(54)

        self.conv2 = nn.Conv2d(34, 15, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(34)

        self.conv3 = nn.Conv2d(15, 2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(15)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.ct1 = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(2)

        self.ct2 = nn.ConvTranspose2d(2, 2, kernel_size=1, stride=1, padding=0)


    def forward(self,x,channel):
        #branch1
        x1=x[:,channel[0]]
        x1 = torch.from_numpy(x1)
        x1 = Variable(x1)
        x1 = F.max_pool2d(F.relu(self.conv1(self.bn1(x1)), inplace=True), kernel_size=2, stride=2)
        x1 = F.max_pool2d(F.relu(self.conv2(self.bn2(x1)), inplace=True), kernel_size=2, stride=2)
        x1 = F.relu(self.conv3(self.bn3(x1)), inplace=True)
        x1 = F.relu(self.ct1(self.up(self.bn4(x1))), inplace=True)
        x1 = F.relu(self.ct2(self.up(self.bn4(x1))))
        x1 = x1.softmax(dim=1)

        #branch2
        x2=x[:,channel[1]]
        x2 = torch.from_numpy(x2)
        x2 = Variable(x2)
        x2 = F.max_pool2d(F.relu(self.conv1(self.bn1(x2)), inplace=True), kernel_size=2, stride=2)
        x2 = F.max_pool2d(F.relu(self.conv2(self.bn2(x2)), inplace=True), kernel_size=2, stride=2)
        x2 = F.relu(self.conv3(self.bn3(x2)), inplace=True)
        x2 = F.relu(self.ct1(self.up(self.bn4(x2))), inplace=True)
        x2 = F.relu(self.ct2(self.up(self.bn4(x2))))
        x2 = x2.softmax(dim=1)

        # branch3
        x3 = x[:, channel[2]]
        x3 = torch.from_numpy(x3)
        x3 = Variable(x3)
        x3 = F.max_pool2d(F.relu(self.conv1(self.bn1(x3)), inplace=True), kernel_size=2, stride=2)
        x3 = F.max_pool2d(F.relu(self.conv2(self.bn2(x3)), inplace=True), kernel_size=2, stride=2)
        x3 = F.relu(self.conv3(self.bn3(x3)), inplace=True)
        x3 = F.relu(self.ct1(self.up(self.bn4(x3))), inplace=True)
        x3 = F.relu(self.ct2(self.up(self.bn4(x3))))
        x3 = x3.softmax(dim=1)

        # branch4
        x4 = x[:, channel[3]]
        x4 = torch.from_numpy(x4)
        x4 = Variable(x4)
        x4 = F.max_pool2d(F.relu(self.conv1(self.bn1(x4)), inplace=True), kernel_size=2, stride=2)
        x4 = F.max_pool2d(F.relu(self.conv2(self.bn2(x4)), inplace=True), kernel_size=2, stride=2)
        x4 = F.relu(self.conv3(self.bn3(x4)), inplace=True)
        x4 = F.relu(self.ct1(self.up(self.bn4(x4))), inplace=True)
        x4 = F.relu(self.ct2(self.up(self.bn4(x4))))
        x4 = x4.softmax(dim=1)

        # branch5
        x5 = x[:, channel[4]]
        x5 = torch.from_numpy(x5)
        x5 = Variable(x5)
        x5 = F.max_pool2d(F.relu(self.conv1(self.bn1(x5)), inplace=True), kernel_size=2, stride=2)
        x5 = F.max_pool2d(F.relu(self.conv2(self.bn2(x5)), inplace=True), kernel_size=2, stride=2)
        x5 = F.relu(self.conv3(self.bn3(x5)), inplace=True)
        x5 = F.relu(self.ct1(self.up(self.bn4(x5))), inplace=True)
        x5 = F.relu(self.ct2(self.up(self.bn4(x5))))
        x5 = x5.softmax(dim=1)

        output=torch.cat((x1,x2,x3,x4,x5),dim=0)
        return output

class TestNet7(nn.Module):
    def __init__(self):
        super(TestNet7,self).__init__()

        self.conv1 = nn.Conv2d(54, 34, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(54)

        self.conv2 = nn.Conv2d(34, 15, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(34)

        self.conv3 = nn.Conv2d(15, 2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(15)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.ct1 = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(2)

        self.ct2 = nn.ConvTranspose2d(2, 2, kernel_size=1, stride=1, padding=0)


    def forward(self,x,channel):
        #branch1
        x1=x[:,channel[0]]
        x1 = torch.from_numpy(x1)
        x1 = Variable(x1)
        x1 = F.max_pool2d(F.relu(self.conv1(self.bn1(x1)), inplace=True), kernel_size=2, stride=2)
        x1 = F.max_pool2d(F.relu(self.conv2(self.bn2(x1)), inplace=True), kernel_size=2, stride=2)
        x1 = F.relu(self.conv3(self.bn3(x1)), inplace=True)
        x1 = F.relu(self.ct1(self.up(self.bn4(x1))), inplace=True)
        x1 = F.relu(self.ct2(self.up(self.bn4(x1))))
        x1 = x1.softmax(dim=1)

        #branch2
        x2=x[:,channel[1]]
        x2 = torch.from_numpy(x2)
        x2 = Variable(x2)
        x2 = F.max_pool2d(F.relu(self.conv1(self.bn1(x2)), inplace=True), kernel_size=2, stride=2)
        x2 = F.max_pool2d(F.relu(self.conv2(self.bn2(x2)), inplace=True), kernel_size=2, stride=2)
        x2 = F.relu(self.conv3(self.bn3(x2)), inplace=True)
        x2 = F.relu(self.ct1(self.up(self.bn4(x2))), inplace=True)
        x2 = F.relu(self.ct2(self.up(self.bn4(x2))))
        x2 = x2.softmax(dim=1)

        # branch3
        x3 = x[:, channel[2]]
        x3 = torch.from_numpy(x3)
        x3 = Variable(x3)
        x3 = F.max_pool2d(F.relu(self.conv1(self.bn1(x3)), inplace=True), kernel_size=2, stride=2)
        x3 = F.max_pool2d(F.relu(self.conv2(self.bn2(x3)), inplace=True), kernel_size=2, stride=2)
        x3 = F.relu(self.conv3(self.bn3(x3)), inplace=True)
        x3 = F.relu(self.ct1(self.up(self.bn4(x3))), inplace=True)
        x3 = F.relu(self.ct2(self.up(self.bn4(x3))))
        x3 = x3.softmax(dim=1)

        # branch4
        x4 = x[:, channel[3]]
        x4 = torch.from_numpy(x4)
        x4 = Variable(x4)
        x4 = F.max_pool2d(F.relu(self.conv1(self.bn1(x4)), inplace=True), kernel_size=2, stride=2)
        x4 = F.max_pool2d(F.relu(self.conv2(self.bn2(x4)), inplace=True), kernel_size=2, stride=2)
        x4 = F.relu(self.conv3(self.bn3(x4)), inplace=True)
        x4 = F.relu(self.ct1(self.up(self.bn4(x4))), inplace=True)
        x4 = F.relu(self.ct2(self.up(self.bn4(x4))))
        x4 = x4.softmax(dim=1)

        # branch5
        x5 = x[:, channel[4]]
        x5 = torch.from_numpy(x5)
        x5 = Variable(x5)
        x5 = F.max_pool2d(F.relu(self.conv1(self.bn1(x5)), inplace=True), kernel_size=2, stride=2)
        x5 = F.max_pool2d(F.relu(self.conv2(self.bn2(x5)), inplace=True), kernel_size=2, stride=2)
        x5 = F.relu(self.conv3(self.bn3(x5)), inplace=True)
        x5 = F.relu(self.ct1(self.up(self.bn4(x5))), inplace=True)
        x5 = F.relu(self.ct2(self.up(self.bn4(x5))))
        x5 = x5.softmax(dim=1)

        # branch6
        x6 = x[:, channel[5]]
        x6 = torch.from_numpy(x6)
        x6 = Variable(x6)
        x6 = F.max_pool2d(F.relu(self.conv1(self.bn1(x6)), inplace=True), kernel_size=2, stride=2)
        x6 = F.max_pool2d(F.relu(self.conv2(self.bn2(x6)), inplace=True), kernel_size=2, stride=2)
        x6 = F.relu(self.conv3(self.bn3(x6)), inplace=True)
        x6 = F.relu(self.ct1(self.up(self.bn4(x6))), inplace=True)
        x6 = F.relu(self.ct2(self.up(self.bn4(x6))))
        x6 = x6.softmax(dim=1)

        # branch7
        x7 = x[:, channel[6]]
        x7 = torch.from_numpy(x7)
        x7 = Variable(x7)
        x7 = F.max_pool2d(F.relu(self.conv1(self.bn1(x7)), inplace=True), kernel_size=2, stride=2)
        x7 = F.max_pool2d(F.relu(self.conv2(self.bn2(x7)), inplace=True), kernel_size=2, stride=2)
        x7 = F.relu(self.conv3(self.bn3(x7)), inplace=True)
        x7 = F.relu(self.ct1(self.up(self.bn4(x7))), inplace=True)
        x7 = F.relu(self.ct2(self.up(self.bn4(x7))))
        x7 = x7.softmax(dim=1)

        output=torch.cat((x1,x2,x3,x4,x5,x6,x7),dim=0)
        return output
    
class TestNet9(nn.Module):
    def __init__(self):
        super(TestNet9,self).__init__()

        self.conv1 = nn.Conv2d(54, 34, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(54)

        self.conv2 = nn.Conv2d(34, 15, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(34)

        self.conv3 = nn.Conv2d(15, 2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(15)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.ct1 = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(2)

        self.ct2 = nn.ConvTranspose2d(2, 2, kernel_size=1, stride=1, padding=0)


    def forward(self,x,channel):
        #branch1
        x1=x[:,channel[0]]
        x1 = torch.from_numpy(x1)
        x1 = Variable(x1)
        x1 = F.max_pool2d(F.relu(self.conv1(self.bn1(x1)), inplace=True), kernel_size=2, stride=2)
        x1 = F.max_pool2d(F.relu(self.conv2(self.bn2(x1)), inplace=True), kernel_size=2, stride=2)
        x1 = F.relu(self.conv3(self.bn3(x1)), inplace=True)
        x1 = F.relu(self.ct1(self.up(self.bn4(x1))), inplace=True)
        x1 = F.relu(self.ct2(self.up(self.bn4(x1))))
        x1 = x1.softmax(dim=1)

        #branch2
        x2=x[:,channel[1]]
        x2 = torch.from_numpy(x2)
        x2 = Variable(x2)
        x2 = F.max_pool2d(F.relu(self.conv1(self.bn1(x2)), inplace=True), kernel_size=2, stride=2)
        x2 = F.max_pool2d(F.relu(self.conv2(self.bn2(x2)), inplace=True), kernel_size=2, stride=2)
        x2 = F.relu(self.conv3(self.bn3(x2)), inplace=True)
        x2 = F.relu(self.ct1(self.up(self.bn4(x2))), inplace=True)
        x2 = F.relu(self.ct2(self.up(self.bn4(x2))))
        x2 = x2.softmax(dim=1)

        # branch3
        x3 = x[:, channel[2]]
        x3 = torch.from_numpy(x3)
        x3 = Variable(x3)
        x3 = F.max_pool2d(F.relu(self.conv1(self.bn1(x3)), inplace=True), kernel_size=2, stride=2)
        x3 = F.max_pool2d(F.relu(self.conv2(self.bn2(x3)), inplace=True), kernel_size=2, stride=2)
        x3 = F.relu(self.conv3(self.bn3(x3)), inplace=True)
        x3 = F.relu(self.ct1(self.up(self.bn4(x3))), inplace=True)
        x3 = F.relu(self.ct2(self.up(self.bn4(x3))))
        x3 = x3.softmax(dim=1)

        # branch4
        x4 = x[:, channel[3]]
        x4 = torch.from_numpy(x4)
        x4 = Variable(x4)
        x4 = F.max_pool2d(F.relu(self.conv1(self.bn1(x4)), inplace=True), kernel_size=2, stride=2)
        x4 = F.max_pool2d(F.relu(self.conv2(self.bn2(x4)), inplace=True), kernel_size=2, stride=2)
        x4 = F.relu(self.conv3(self.bn3(x4)), inplace=True)
        x4 = F.relu(self.ct1(self.up(self.bn4(x4))), inplace=True)
        x4 = F.relu(self.ct2(self.up(self.bn4(x4))))
        x4 = x4.softmax(dim=1)

        # branch5
        x5 = x[:, channel[4]]
        x5 = torch.from_numpy(x5)
        x5 = Variable(x5)
        x5 = F.max_pool2d(F.relu(self.conv1(self.bn1(x5)), inplace=True), kernel_size=2, stride=2)
        x5 = F.max_pool2d(F.relu(self.conv2(self.bn2(x5)), inplace=True), kernel_size=2, stride=2)
        x5 = F.relu(self.conv3(self.bn3(x5)), inplace=True)
        x5 = F.relu(self.ct1(self.up(self.bn4(x5))), inplace=True)
        x5 = F.relu(self.ct2(self.up(self.bn4(x5))))
        x5 = x5.softmax(dim=1)

        # branch6
        x6 = x[:, channel[5]]
        x6 = torch.from_numpy(x6)
        x6 = Variable(x6)
        x6 = F.max_pool2d(F.relu(self.conv1(self.bn1(x6)), inplace=True), kernel_size=2, stride=2)
        x6 = F.max_pool2d(F.relu(self.conv2(self.bn2(x6)), inplace=True), kernel_size=2, stride=2)
        x6 = F.relu(self.conv3(self.bn3(x6)), inplace=True)
        x6 = F.relu(self.ct1(self.up(self.bn4(x6))), inplace=True)
        x6 = F.relu(self.ct2(self.up(self.bn4(x6))))
        x6 = x6.softmax(dim=1)

        # branch7
        x7 = x[:, channel[6]]
        x7 = torch.from_numpy(x7)
        x7 = Variable(x7)
        x7 = F.max_pool2d(F.relu(self.conv1(self.bn1(x7)), inplace=True), kernel_size=2, stride=2)
        x7 = F.max_pool2d(F.relu(self.conv2(self.bn2(x7)), inplace=True), kernel_size=2, stride=2)
        x7 = F.relu(self.conv3(self.bn3(x7)), inplace=True)
        x7 = F.relu(self.ct1(self.up(self.bn4(x7))), inplace=True)
        x7 = F.relu(self.ct2(self.up(self.bn4(x7))))
        x7 = x7.softmax(dim=1)

        # branch8
        x8 = x[:, channel[7]]
        x8 = torch.from_numpy(x8)
        x8 = Variable(x8)
        x8 = F.max_pool2d(F.relu(self.conv1(self.bn1(x8)), inplace=True), kernel_size=2, stride=2)
        x8 = F.max_pool2d(F.relu(self.conv2(self.bn2(x8)), inplace=True), kernel_size=2, stride=2)
        x8 = F.relu(self.conv3(self.bn3(x8)), inplace=True)
        x8 = F.relu(self.ct1(self.up(self.bn4(x8))), inplace=True)
        x8 = F.relu(self.ct2(self.up(self.bn4(x8))))
        x8 = x8.softmax(dim=1)

        # branch9
        x9 = x[:, channel[8]]
        x9 = torch.from_numpy(x9)
        x9 = Variable(x9)
        x9 = F.max_pool2d(F.relu(self.conv1(self.bn1(x9)), inplace=True), kernel_size=2, stride=2)
        x9 = F.max_pool2d(F.relu(self.conv2(self.bn2(x9)), inplace=True), kernel_size=2, stride=2)
        x9 = F.relu(self.conv3(self.bn3(x9)), inplace=True)
        x9 = F.relu(self.ct1(self.up(self.bn4(x9))), inplace=True)
        x9 = F.relu(self.ct2(self.up(self.bn4(x9))))
        x9 = x9.softmax(dim=1)

        output=torch.cat((x1,x2,x3,x4,x5,x6,x7,x8,x9),dim=0)
        return output

class TestNet11(nn.Module):
    def __init__(self):
        super(TestNet11,self).__init__()

        self.conv1 = nn.Conv2d(54, 34, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(54)

        self.conv2 = nn.Conv2d(34, 15, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(34)

        self.conv3 = nn.Conv2d(15, 2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(15)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.ct1 = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(2)

        self.ct2 = nn.ConvTranspose2d(2, 2, kernel_size=1, stride=1, padding=0)


    def forward(self,x,channel):
        #branch1
        x1=x[:,channel[0]]
        x1 = torch.from_numpy(x1)
        x1 = Variable(x1)
        x1 = F.max_pool2d(F.relu(self.conv1(self.bn1(x1)), inplace=True), kernel_size=2, stride=2)
        x1 = F.max_pool2d(F.relu(self.conv2(self.bn2(x1)), inplace=True), kernel_size=2, stride=2)
        x1 = F.relu(self.conv3(self.bn3(x1)), inplace=True)
        x1 = F.relu(self.ct1(self.up(self.bn4(x1))), inplace=True)
        x1 = F.relu(self.ct2(self.up(self.bn4(x1))))
        x1 = x1.softmax(dim=1)

        #branch2
        x2=x[:,channel[1]]
        x2 = torch.from_numpy(x2)
        x2 = Variable(x2)
        x2 = F.max_pool2d(F.relu(self.conv1(self.bn1(x2)), inplace=True), kernel_size=2, stride=2)
        x2 = F.max_pool2d(F.relu(self.conv2(self.bn2(x2)), inplace=True), kernel_size=2, stride=2)
        x2 = F.relu(self.conv3(self.bn3(x2)), inplace=True)
        x2 = F.relu(self.ct1(self.up(self.bn4(x2))), inplace=True)
        x2 = F.relu(self.ct2(self.up(self.bn4(x2))))
        x2 = x2.softmax(dim=1)

        # branch3
        x3 = x[:, channel[2]]
        x3 = torch.from_numpy(x3)
        x3 = Variable(x3)
        x3 = F.max_pool2d(F.relu(self.conv1(self.bn1(x3)), inplace=True), kernel_size=2, stride=2)
        x3 = F.max_pool2d(F.relu(self.conv2(self.bn2(x3)), inplace=True), kernel_size=2, stride=2)
        x3 = F.relu(self.conv3(self.bn3(x3)), inplace=True)
        x3 = F.relu(self.ct1(self.up(self.bn4(x3))), inplace=True)
        x3 = F.relu(self.ct2(self.up(self.bn4(x3))))
        x3 = x3.softmax(dim=1)

        # branch4
        x4 = x[:, channel[3]]
        x4 = torch.from_numpy(x4)
        x4 = Variable(x4)
        x4 = F.max_pool2d(F.relu(self.conv1(self.bn1(x4)), inplace=True), kernel_size=2, stride=2)
        x4 = F.max_pool2d(F.relu(self.conv2(self.bn2(x4)), inplace=True), kernel_size=2, stride=2)
        x4 = F.relu(self.conv3(self.bn3(x4)), inplace=True)
        x4 = F.relu(self.ct1(self.up(self.bn4(x4))), inplace=True)
        x4 = F.relu(self.ct2(self.up(self.bn4(x4))))
        x4 = x4.softmax(dim=1)

        # branch5
        x5 = x[:, channel[4]]
        x5 = torch.from_numpy(x5)
        x5 = Variable(x5)
        x5 = F.max_pool2d(F.relu(self.conv1(self.bn1(x5)), inplace=True), kernel_size=2, stride=2)
        x5 = F.max_pool2d(F.relu(self.conv2(self.bn2(x5)), inplace=True), kernel_size=2, stride=2)
        x5 = F.relu(self.conv3(self.bn3(x5)), inplace=True)
        x5 = F.relu(self.ct1(self.up(self.bn4(x5))), inplace=True)
        x5 = F.relu(self.ct2(self.up(self.bn4(x5))))
        x5 = x5.softmax(dim=1)

        # branch6
        x6 = x[:, channel[5]]
        x6 = torch.from_numpy(x6)
        x6 = Variable(x6)
        x6 = F.max_pool2d(F.relu(self.conv1(self.bn1(x6)), inplace=True), kernel_size=2, stride=2)
        x6 = F.max_pool2d(F.relu(self.conv2(self.bn2(x6)), inplace=True), kernel_size=2, stride=2)
        x6 = F.relu(self.conv3(self.bn3(x6)), inplace=True)
        x6 = F.relu(self.ct1(self.up(self.bn4(x6))), inplace=True)
        x6 = F.relu(self.ct2(self.up(self.bn4(x6))))
        x6 = x6.softmax(dim=1)

        # branch7
        x7 = x[:, channel[6]]
        x7 = torch.from_numpy(x7)
        x7 = Variable(x7)
        x7 = F.max_pool2d(F.relu(self.conv1(self.bn1(x7)), inplace=True), kernel_size=2, stride=2)
        x7 = F.max_pool2d(F.relu(self.conv2(self.bn2(x7)), inplace=True), kernel_size=2, stride=2)
        x7 = F.relu(self.conv3(self.bn3(x7)), inplace=True)
        x7 = F.relu(self.ct1(self.up(self.bn4(x7))), inplace=True)
        x7 = F.relu(self.ct2(self.up(self.bn4(x7))))
        x7 = x7.softmax(dim=1)

        # branch8
        x8 = x[:, channel[7]]
        x8 = torch.from_numpy(x8)
        x8 = Variable(x8)
        x8 = F.max_pool2d(F.relu(self.conv1(self.bn1(x8)), inplace=True), kernel_size=2, stride=2)
        x8 = F.max_pool2d(F.relu(self.conv2(self.bn2(x8)), inplace=True), kernel_size=2, stride=2)
        x8 = F.relu(self.conv3(self.bn3(x8)), inplace=True)
        x8 = F.relu(self.ct1(self.up(self.bn4(x8))), inplace=True)
        x8 = F.relu(self.ct2(self.up(self.bn4(x8))))
        x8 = x8.softmax(dim=1)

        # branch9
        x9 = x[:, channel[8]]
        x9 = torch.from_numpy(x9)
        x9 = Variable(x9)
        x9 = F.max_pool2d(F.relu(self.conv1(self.bn1(x9)), inplace=True), kernel_size=2, stride=2)
        x9 = F.max_pool2d(F.relu(self.conv2(self.bn2(x9)), inplace=True), kernel_size=2, stride=2)
        x9 = F.relu(self.conv3(self.bn3(x9)), inplace=True)
        x9 = F.relu(self.ct1(self.up(self.bn4(x9))), inplace=True)
        x9 = F.relu(self.ct2(self.up(self.bn4(x9))))
        x9 = x9.softmax(dim=1)

        # branch10
        x10 = x[:, channel[9]]
        x10 = torch.from_numpy(x10)
        x10 = Variable(x10)
        x10 = F.max_pool2d(F.relu(self.conv1(self.bn1(x10)), inplace=True), kernel_size=2, stride=2)
        x10 = F.max_pool2d(F.relu(self.conv2(self.bn2(x10)), inplace=True), kernel_size=2, stride=2)
        x10 = F.relu(self.conv3(self.bn3(x10)), inplace=True)
        x10 = F.relu(self.ct1(self.up(self.bn4(x10))), inplace=True)
        x10 = F.relu(self.ct2(self.up(self.bn4(x10))))
        x10 = x10.softmax(dim=1)

        # branch11
        x11 = x[:, channel[10]]
        x11 = torch.from_numpy(x11)
        x11 = Variable(x11)
        x11 = F.max_pool2d(F.relu(self.conv1(self.bn1(x11)), inplace=True), kernel_size=2, stride=2)
        x11 = F.max_pool2d(F.relu(self.conv2(self.bn2(x11)), inplace=True), kernel_size=2, stride=2)
        x11 = F.relu(self.conv3(self.bn3(x11)), inplace=True)
        x11 = F.relu(self.ct1(self.up(self.bn4(x11))), inplace=True)
        x11 = F.relu(self.ct2(self.up(self.bn4(x11))))
        x11 = x11.softmax(dim=1)

        output=torch.cat((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11),dim=0)
        return output
    
class TestNetn(nn.Module):
    def __init__(self,input_channels,branch_num,branch_layer):
        super(TestNetn,self).__init__()

        self.input_channels=input_channels
        self.branch_num=branch_num
        self.branch_layer=branch_layer

        self.conv1 = nn.Conv2d(self.input_channels, 34, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.input_channels)

        self.conv2 = nn.Conv2d(34, 15, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(34)

        self.conv3 = nn.Conv2d(15, 2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(15)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.ct1 = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(2)

        self.ct2 = nn.ConvTranspose2d(2, 2, kernel_size=1, stride=1, padding=0)


    def forward(self,x,channel):
        #branch
        for i in range(self.branch_num):
            x1=x[:,channel[i]]
            x1 = torch.from_numpy(x1)
            x1 = Variable(x1)
            x1 = F.max_pool2d(F.relu(self.conv1(self.bn1(x1)), inplace=True), kernel_size=2, stride=2)
            x1 = F.max_pool2d(F.relu(self.conv2(self.bn2(x1)), inplace=True), kernel_size=2, stride=2)
            x1 = F.relu(self.conv3(self.bn3(x1)), inplace=True)
            x1 = F.relu(self.ct1(self.up(self.bn4(x1))), inplace=True)
            x1 = F.relu(self.ct2(self.up(self.bn4(x1))))
            x1 = x1.softmax(dim=1)
            if i!=0:
                output=torch.cat((output,x1),dim=0)
            else:
                output=x1

        return output


