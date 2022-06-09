import glob

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_aff2_train_data():
    '''
    data loader
    '''
    list_csv = glob.glob("/data/users/ys221/data/ABAW/labels_save/expression/Train_Set_v1/" + '*')
    df = pd.DataFrame()
    for i in list_csv:
        df = pd.concat((df, pd.read_csv(i)), axis=0).reset_index(drop=True)
    list_labels_ex = np.array(df['labels_ex'].values)
    list_image_id = np.array(df['image_id'].values)

    images = []
    labels = []
    idn = []
    # L1 = random.sample(range(1, 421093), 1000)
    for i in tqdm(range(2000)):
        j = i * 40
        image = cv2.imread(list_image_id[j])[..., ::-1].tolist()
        # image = cv2.resize(image, [64, 64], interpolation=cv2.INTER_AREA).tolist()
        images.append(image)
        labels.append(list_labels_ex[j])
        seq = list_image_id[j].split('/')[7]
        # idn.append(seq.split('-')[0])
    images = np.stack(images)
    images = images.astype(float)
    labels = np.stack(labels)

    return images, labels, idn


def get_aff2_valid_data():
    '''
    data loader
    '''
    list_csv = glob.glob("/data/users/ys221/data/ABAW/labels_save/expression/Validation_Set_v1/" + '*')
    df = pd.DataFrame()
    for i in list_csv:
        df = pd.concat((df, pd.read_csv(i)), axis=0).reset_index(drop=True)
    list_labels_ex = np.array(df['labels_ex'].values)
    list_image_id = np.array(df['image_id'].values)

    images = []
    labels = []
    idn = []
    # L1 = random.sample(range(1, 221093), 1000)
    for i in tqdm(range(2000)):
        j = i * 40
        image = cv2.imread(list_image_id[j])[..., ::-1].tolist()
        # image = cv2.resize(image, [64, 64], interpolation=cv2.INTER_AREA).tolist()
        images.append(image)
        labels.append(list_labels_ex[j])
        seq = list_image_id[j].split('/')[7]
        idn.append(seq.split('-')[0])
    images = np.stack(images)
    images = images.astype(float)
    labels = np.stack(labels)

    return images, labels, idn


def get_VoxCeleb1_data():

    return

def get_VoxCeleb2_data():

    return

# class get_data:
#     def __init__(self, dataset_list):
#         images_all = []
#         labels_all = []
#         for dataset in dataset_list:
#             if dataset == 'aff2_train':
#                 images, labels, idn = get_aff2_train_data()
#                 images_all.append(images)
#                 labels_all.append(labels)
#             if dataset == 'aff2_valid':
#                 images, labels, idn = get_aff2_valid_data()
#                 images_all.append(images)
#                 labels_all.append(labels)
#             if dataset == 'VoxCeleb1':
#                 images, labels, idn = get_VoxCeleb1_data()
#                 images_all.append(images)
#                 labels_all.append(labels)
#             if dataset == 'VoxCeleb1':
#                 images, labels, idn = get_VoxCeleb2_data()
#                 images_all.append(images)
#                 labels_all.append(labels)




