import random
import numpy as np
import cv2
import torch
import random

class BinaryMapError(ValueError):
    def __init__(self,value):
        self.value=value

    def __str__(self):
        return repr(self.value)



class Testlib():
    def __init__(self,num):
        self.data=[]
        self.channels=[]
        self.branch_No=[]
        self.branch_net=[]
        self.net_num=num

    def add_data(self,data):
        self.data=data

    def add_channel(self,channel):
        self.channels.append(channel)

    def get_branch_no(self):
        for i in range(self.net_num):
            self.branch_No.append(random.randint(1,self.net_num))




class process():
    def __init__(self):
        self.image_train_list=[]
        self.image_test_list=[]
        self.channels=[]
        self.image=[]
        self.max=255

    # def generate_imageList(self):
    #     self.set_parameters(net_name="End_to_End_L5_BN1", No=1, epoch=1, file_name="0001.mat",
    #                           first_level_path="dataset")
    #     self.is_file(self.dir_name(), mode="second")
    #     self.setcwd(self.current_file)

    def get_imageList(self,lists):
        lists.sort()
        self.image_train_list = lists[0::2]
        self.image_test_list = lists[1::2]
        imagelist = lists[0::2]
        imagelist.insert(0, "train_list:")
        imagelist.append("test_list:")
        imagelist.extend(self.image_test_list)
        return "\n".join(imagelist)

    def read_imageList(self,data):
        data="".join(data)
        data=data.split("test_list:\n")
        if "train_list" in data[0]:
            self.image_train_list=data[0].split("\n")
            self.image_train_list=self.image_train_list[1:-1]
            self.image_test_list=data[-1].split("\n")
        elif "train_list" in data[-1]:
            data = data[-1].split("\ntrain_list:\n")
            self.image_test_list = data[0]
            self.image_train_list = data[1]

    def get_sample(self,samples,branch_num):
        lists = list(range(0, 81))
        channels=[]
        for i in range(branch_num):
            channel_list = random.sample(lists, samples)
            channel_list.sort()
            channels.append(channel_list)
        return channels

    def img_process(self,img):
        img=img["hypercube"][()].astype('float32')
        img=np.swapaxes(img,1,2)
        image_data = np.expand_dims(img, axis=0)  # shape: [1, 81, 1024, 768]
        data = image_data / self.max
        return data

    def img_process_with_jpg(self,img):
        max_value=1
        if img.max()<=1:
            max_value=1
        elif img.max()<=255:
            max_value=255
        else:
            max_value=self.max

        try:
            image=cv2.threshold(img,max_value/2,1,cv2.THRESH_BINARY)
            image=image[1]
            image2=img/max_value
            temp=image2-image.astype("float32")
            if temp.max()>0:
                raise BinaryMapError(temp.max)

        except BinaryMapError as e:
            print("BinaryMapError:Expected Binary Map but got Map with Multiple value,max to ",e.value())
            print("Input Map has been transfered to Binary Map")
            pass

        return image

    def saliency_process(self,saliency):
        saliency=saliency[-1]
        img=saliency.detach().numpy()
        img=img[0]*255
        return img[0]

    def eva_process(self,saliency,ground_truth):
        saliency = saliency[1].contiguous().view(1, -1).permute(1, 0)
        saliency = saliency.detach().numpy()
        ground_truth=torch.from_numpy(ground_truth)
        ground_truth = ground_truth.contiguous().view(-1, 1).float()
        ground_truth = ground_truth.detach().numpy()
        return saliency,ground_truth

    def eva_process_with_jpg(self,saliency,ground_truth):
        saliency = saliency.reshape(-1,1)
        ground_truth = ground_truth.reshape(-1,1)
        return saliency,ground_truth


    def ground_truth_process(self,gt):
        gt = cv2.threshold(gt, 255 / 2, 1, cv2.THRESH_BINARY)

        return gt[1]

    def convert(self,func,seq):

        return [func(item) for item in seq]

    def num2int(self,seq):
        return self.convert(int,seq)

    def num2float(self,seq):
        return self.convert(float,seq)
