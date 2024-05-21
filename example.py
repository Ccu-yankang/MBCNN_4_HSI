from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pysaliency
import cv2

path="D:\\workspace\\python_workspace\\MBCNN\\MBCNN\\net\\End_to_End_L5_BN1\\End_to_End_L5_BN1_No.2\\result\epoch1\\0051.jpg"
img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)

img1=cv2.threshold(img,255/2,1,cv2.THRESH_BINARY)
img1=255*img1[1].astype("int16")

x1=np.random.randint(0,100,(10,10))
x2=np.random.randint(0,100,(10,5))
p=stats.pearsonr(x1,x2)
print(p)


dataset_location = 'datasets'
model_location = 'models'

mit_stimuli, mit_fixations = pysaliency.external_datasets.get_mit1003(location=dataset_location)
aim = pysaliency.AIM(location=model_location)
saliency_map = aim.saliency_map(mit_stimuli.stimuli[0])

plt.imshow(saliency_map)


auc = aim.AUC(mit_stimuli, mit_fixations)