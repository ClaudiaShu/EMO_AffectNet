

########################## first part: prepare data ###########################
from random import shuffle
import glob
import os

import pandas as pd


shuffle_data = True  # shuffle the addresses

prefix = "/mnt/c/Data/Yuxuan/AffectNet"
hdf5_file_train = 'AffectNet_train.hdf5'
hdf5_path_train = os.path.join(prefix, hdf5_file_train)  # file path for the created .hdf5 file

hdf5_file_valid = 'AffectNet_valid.hdf5'
hdf5_path_valid = os.path.join(prefix, hdf5_file_valid)

image_train_path = os.path.join(prefix, 'train_set', 'images')  # the original data path
image_val_path = os.path.join(prefix, 'val_set', 'images')
# get all the image paths
# addrs = glob.glob(image_train_path)

# Get train
train_file = 'train_set_data.csv'
df_file = os.path.join(prefix, train_file)
df_train = pd.read_csv(df_file)

train_addrs = df_train['image_id'].values
train_arousal = df_train['labels_aro'].values
train_valence = df_train['labels_val'].values
train_labels = df_train['labels_exp'].values

# shuffle data
if shuffle_data:
    c = list(zip(train_addrs, train_arousal, train_valence, train_labels))  # use zip() to bind the images and labels together
    shuffle(c)

    (train_addrs, train_arousal, train_valence, train_labels) = zip(*c)  # *c is used to separate all the tuples in the list c,
    # "addrs" then contains all the shuffled paths and
    # "labels" contains all the shuffled labels.

# Get valid
valid_file = 'val_set_data.csv'
df_file = os.path.join(prefix, valid_file)
df_valid = pd.read_csv(df_file)

valid_addrs = df_valid['image_id'].values
valid_arousal = df_valid['labels_aro'].values
valid_valence = df_valid['labels_val'].values
valid_labels = df_valid['labels_exp'].values

# shuffle data
if shuffle_data:
    c = list(zip(valid_addrs, valid_arousal, valid_valence, valid_labels))  # use zip() to bind the images and labels together
    shuffle(c)

    (valid_addrs, valid_arousal, valid_valence, valid_labels) = zip(*c)  # *c is used to separate all the tuples in the list c,
    # "addrs" then contains all the shuffled paths and
    # "labels" contains all the shuffled labels.

# # Divide the data into 80% for train and 20% for valid
# train_addrs = addrs[0:int(0.8 * len(addrs))]
# train_labels = labels[0:int(0.8 * len(labels))]
# 
# valid_addrs = addrs[int(0.8 * len(addrs)):]
# valid_labels = labels[int(0.8 * len(labels)):]

##################### second part: create the h5py object #####################
import numpy as np
import h5py

######################## third part: write the images #########################
import cv2
from tqdm import tqdm

train_shape = (len(train_addrs), 112, 112, 3)
valid_shape = (len(valid_addrs), 112, 112, 3)

# Valid
f2 = h5py.File(hdf5_path_valid, mode='w')
f2.create_dataset("imgs", valid_shape, np.uint8)

f2.create_dataset("labels", (len(valid_addrs),), np.uint8)
f2["labels"][...] = valid_labels

f2.create_dataset("arousal", (len(valid_addrs),), np.float32)
f2["arousal"][...] = valid_arousal

f2.create_dataset("valence", (len(valid_addrs),), np.float32)
f2["valence"][...] = valid_valence


# loop over valid paths
for i in tqdm(range(len(valid_addrs))):

    # if i % 1000 == 0 and i > 1:
    #     print('Valid data: {}/{}'.format(i, len(valid_addrs)))

    addr = valid_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    f2["imgs"][i, ...] = img[None]

f2.close()


# open a hdf5 file and create earrays
f1 = h5py.File(hdf5_path_train, mode='w')

# PIL.Image: the pixels range is 0-255,dtype is uint.
# matplotlib: the pixels range is 0-1,dtype is float.
f1.create_dataset("imgs", train_shape, np.uint8)

# the ".create_dataset" object is like a dictionary, the "train_labels" is the key.
# Train
f1.create_dataset("labels", (len(train_addrs),), np.uint8)
f1["labels"][...] = train_labels

f1.create_dataset("arousal", (len(train_addrs),), np.float32)
f1["arousal"][...] = train_arousal

f1.create_dataset("valence", (len(train_addrs),), np.float32)
f1["valence"][...] = train_valence

# loop over train paths
for i in tqdm(range(len(train_addrs))):

    # if i % 1000 == 0 and i > 1:
    #     print('Train data: {}/{}'.format(i, len(train_addrs)))

    addr = train_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_CUBIC)  # resize to (112,112)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 load images as BGR, convert it to RGB
    f1["imgs"][i, ...] = img[None]

f1.close()

