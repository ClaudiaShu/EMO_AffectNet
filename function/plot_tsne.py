import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch import nn

from function.plot_utils import get_aff2_train_data
from models import *
# Experimental: HDBScan is a state-of-the-art clustering algorithm
from models.model_RES import RES_SSL
from models.model_RES_imagenet import RES_feature

hdbscan_available = True
try:
    import hdbscan
except ImportError:

    hdbscan_available = False

# device = "cuda:0" if torch.cuda.is_available() else 'cpu'
# os.environ['CUDA_VISIBLE_DEVICES']='1'
device = 'cpu'


'''
Data
'''
images, labels, idn = get_aff2_train_data()

'''
models
'''

model = RES_feature().to(device)
model.out = nn.Sequential()

model.to(device)
# '''

images = torch.from_numpy(images).to(device=device, dtype=torch.float)
# labels = torch.from_numpy(labels).to(device=device, dtype=torch.long)
print(f'type of labels:{labels.dtype},type of images:{images.dtype}')
# images = torch.unsqueeze(images, dim=2)
print(f'shape of labels:{labels.shape}, shape of images:{images.shape}')
images = images.permute(0, 3, 1, 2)
X = model(images).cpu().detach().numpy()
print(f'feature shape:{X.shape}')
# X_std = StandardScaler().fit_transform(X)
y = labels

# tsne = TSNE(n_components= 2, perplexity= 50, verbose=2)
tsne = TSNE(n_components=2, perplexity=100, learning_rate=1000, n_iter=1000, random_state=0)
X_tsne = tsne.fit_transform(X)
# print(X_tsne.shape)#batchsize, n_components
# plt.scatter(X_tsne[:,0],X_tsne[:,1])

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(20, 20))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(idn[i]), color=plt.cm.Set1(y[i]),
             fontdict={'weight': 'bold', 'size': 8})

plt.xticks([])
plt.yticks([])
plt.show()


