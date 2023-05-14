import os
import warnings

import pandas as pd
import numpy as np
import numpy.linalg as linalg
import matplotlib.pylab as plt

import torch
from torch import nn
import torchvision
import torchvision.transforms as T

from sklearn.datasets import fetch_lfw_people
from tqdm import tqdm

from utils import EXPR_metric

lfw_dataset = fetch_lfw_people(min_faces_per_person=200) # To keep people who got 200 images or more
import time as time # To calculate time execution
from sklearn.model_selection import train_test_split # To split data
from sklearn.metrics import confusion_matrix,classification_report

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
warnings.filterwarnings("ignore")
device = "cuda:0" if torch.cuda.is_available() else 'cpu'

img_height = 62 # Images height in pixels
img_width = 47 # Images width in pixels
X = lfw_dataset.data # Line vector of image grey levels
y = lfw_dataset.target # Images labels
names = lfw_dataset.target_names # Peoples name
print(lfw_dataset.images.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def get_transform(size=46):
    mu, st = 0, 255
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.ToTensor(),
            T.RandomCrop(size=(size, size)),
            T.Normalize(mean=(mu,), std=(st,))
        ]
    )
    return transform

class RES_SSL(nn.Module):
    def __init__(self, model_dir, dim=128):
        super(RES_SSL, self).__init__()
        dir = model_dir
        # self.pretrain = torchvision.models.resnet50(pretrained=True).cuda()
        self.pretrain = torchvision.models.resnet18(pretrained=True).cuda()
       
        self.pretrain.fc = nn.Identity()
        # self.pretrain.load_state_dict(state_dict, strict=False)
        self.pretrain.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        self.pretrain.eval()
        return self.pretrain(x)

def get_model():
    model_state_dict = "/data2/ys221/data/model/SSL_model/swap/checkpoint_0410.pth.tar"
    model = RES_SSL(model_dir=model_state_dict)
    return model

def distance(I1,I2):
    return np.sqrt(sum((I1-I2)**2))

n_test = X_test.shape[0] # number of individuals in the test sample
n_train = X_train.shape[0] # number of individuals in the train sample
y_pred = np.zeros(n_test) # Initialization of the predictions vector
t1=time.time()
timesArray = np.array([])
size = 46
transform = get_transform(size)
model = get_model()
model.to(device)

model.eval()
# with torch.no_grad:

X_test_img = X_test.reshape((X_test.shape[0],img_height, img_width))
X_train_img = X_train.reshape((X_train.shape[0],img_height, img_width))
# np.expand_dims(

XX_test = []
XX_train = []

for image in X_test_img:
    img_trans = transform(image)
    XX_test.append(img_trans)
XX_test = np.expand_dims(np.concatenate(XX_test), axis=1)
XX_test_fea = model(torch.from_numpy(XX_test).to(device)).detach().cpu().numpy()

for image in X_train_img:
    img_trans = transform(image)
    XX_train.append(img_trans)
XX_train = np.expand_dims(np.concatenate(XX_train), axis=1)
XX_train_fea = model(torch.from_numpy(XX_train).to(device)).detach().cpu().numpy()


for i in tqdm(range(n_test)):
    mini_index = 0
    for j in range(n_train):
        if (distance(XX_test_fea[i, :], XX_train_fea[j, :]) < distance(XX_test_fea[i, :], XX_train_fea[mini_index, :])):
            mini_index = j
    y_pred[i] = y_train[mini_index]
t2=time.time()
print(f'The time execution of the KNN Algorithm with the X_test and the X_train matrix and the distance function is : {t2-t1} seconds')
timesArray = np.append(timesArray,t2-t1)

def cm_summary(test,prediction):
    """ Builds a plot of the confusion matrix and the report associated with it
    :param test: The test array
    :type test: numpy.ndarray
    :param prediction: The prediction array
    :type prediction: numpy.ndarray
    :returns: The graphical representation of the confusion matrix and the report associated with it
    :rtype: matplotlib.image.AxesImage, str
    """
    cm = confusion_matrix(test,prediction)
    matrix = plt.matshow(cm)
    report = classification_report(test, prediction)
    return matrix,report

matrix,report = cm_summary(y_test,y_pred)
print(report)
f1, acc, total = EXPR_metric(y_pred, y_test)
print(f'f1:{f1}; Accuracy:{acc}')

