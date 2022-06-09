import glob

import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_aug import get_image_transform
from data_loader import Aff2_Dataset_static_shuffle
from function.plot_utils import get_data
from models.model_RES import RES_SSL
from utils import create_original_data

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

'''
Data
'''
images, labels, idn = get_data()

# load models and features here
model = RES_SSL().to(device)
model.out = nn.Sequential()
rep = model(images)
rep = rep.cpu().detach()

# PCA
np_rep = rep.numpy()
pca = PCA()
pca.fit(np_rep)
# pca.explained_variance_ratio_

from mpl_toolkits.mplot3d import Axes3D

X = rep[:, 0].numpy()
Y = rep[:, 1].numpy()
Z = rep[:, 2].numpy()

# 3D Graph
fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(X, Y, Z, c=labels, marker="x", cmap='rainbow')
plt.show()