import torch
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
from pathlib import Path
import cv2

# c_path=".\\dataset\\HS-SOD\\ground_truth"
# path=Path(c_path)
#
# files=path.glob("*.jpg")
# print(files)

class Data_set(Dataset):
    def __init__(self,train_path,test_path):
        self.data_path=[]
        self.gt_path=[]
        paths=Path(train_path)
        for file in paths:
            self.data_path.append(file)

        paths=Path(test_path)
        for file in paths:
            self.gt_path.append(file)

    def __getitem__(self, index):
        img=h5py.File(self.data_path[index])
        gt=cv2.imread(self.gt_path[index],cv2.IMREAD_GRAYSCALE)
        return img,gt

    def __len__(self):
        return len(self.data_path)

class Module_set(Dataset):
    def __init__(self,location):

        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
