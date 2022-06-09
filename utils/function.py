import glob
import os
import random
from torch.autograd import Variable
import PIL.Image as Image
import torch
from sklearn.model_selection import GroupKFold, KFold
import numpy as np
import pandas as pd
import torch
import cv2

from tqdm import *

def read_txt(txt_file):
    with open(txt_file, 'r') as f:
        videos = f.readlines()
    videos = [x.strip() for x in videos]
    return videos

def seed_everything(seed=1):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def check_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def adjust_learning_rate(optimizer, epoch, milestones=None):
    """Sets the learning rate: milestone is a list/tuple"""

    def to(epoch):
        if epoch <= args.warmup:
            return 1
        elif args.warmup < epoch <= milestones[0]:
            return 0
        for i in range(1, len(milestones)):
            if milestones[i - 1] < epoch <= milestones[i]:
                return i
        return len(milestones)

    n = to(epoch)

    global lr
    lr = args.base_lr * (0.2 ** n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def unfreeze(model, percent=0.25):
    l = int(np.ceil(len(model._modules.keys()) * percent))
    l = list(model._modules.keys())[-l:]
    print(f"unfreezing these layer {l}", )
    for name in l:
        for params in model._modules[name].parameters():
            params.requires_grad_(True)


def pad_if_need(df, length):
    df = df.sort_values(by=['labels_ex', 'image_id'])
    df.labels_ex.value_counts()
    out_df = pd.DataFrame()
    for i in df.labels_ex.unique():
        df1 = df[df.labels_ex == i]
        df1 = df1.append(df1.iloc[[-1] * (length - len(df1) % length)]).reset_index(drop=True)
        out_df = pd.concat([out_df, df1], axis=0).reset_index(drop=True)
    return out_df


def ensemble_ex(root):
    list_np = glob.glob(root + '/*')
    results = []
    for i in list_np:
        result = np.load(i)
        results.append(result)
    results = np.stack(results, axis=0)
    # import pdb; pdb.set_trace()
    results = np.mean(results, axis=0)
    results = np.argmax(results, axis=1)
    return results


def get_one_hot(label, num_classes):
    batch_size = label.shape[0]
    onehot_label = torch.zeros((batch_size, num_classes))
    onehot_label = onehot_label.scatter_(1, label.unsqueeze(1).detach().cpu(), 1)
    onehot_label = (onehot_label.type(torch.FloatTensor)).to(label.device)
    return onehot_label


def to_onehot_ex(label):
    arr = torch.zeros(7)
    arr[label] = 1
    return arr


def accuracy(output, label):
    cnt = label.shape[0]
    true_count = (output == label).sum()
    now_accuracy = true_count / cnt
    return now_accuracy, cnt


def mixup(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def create_original_data(file_name):
    list_csv = glob.glob(file_name)
    df = pd.DataFrame()
    for i in tqdm(list_csv, total=len(list_csv)):
        df = pd.concat((df, pd.read_csv(i)), axis=0).reset_index(drop=True)
    return df


def create_balance_data_ex(file_name):
    list_csv = glob.glob(file_name)

    df = pd.DataFrame()
    for i in tqdm(list_csv):
        df = pd.concat((df, pd.read_csv(i, index_col=0)), axis=0).reset_index(drop=True)

    df3 = df
    # TODO maybe change the rate of sample
    weights = df3.labels_ex.value_counts().sort_values()[0] / df3.labels_ex.value_counts().sort_values()

    df4 = pd.DataFrame()
    for k, v in dict(weights).items():
        df4 = pd.concat([df4, df3[df3['labels_ex'] == k].sample(frac=v, random_state=1, replace=True)],
                        axis=0).reset_index(drop=True)

    return df4


def create_test_df(path_txt, out_path):
    au = open(path_txt, 'r')
    lines = au.readlines()
    au_video = []
    for line in lines:
        au_video.append(line.strip())

    k = []
    v = []
    for video_dir in list(au_video):
        k.append(video_dir)
        if 'left' in video_dir:
            video_dir = video_dir.replace('_left', '')
        elif 'right' in video_dir:
            video_dir = video_dir.replace('_right', '')
        if os.path.exists('../all_videos/' + video_dir + '.mp4'):
            path = '../all_videos/' + video_dir + '.mp4'
        else:
            path = '../all_videos/' + video_dir + '.avi'

        v_cap = cv2.VideoCapture(path)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v.append(v_len)

    df1 = pd.DataFrame(columns=['name_videos'], data=k)
    df1['len_images'] = v

    au_video = ['/data/users/ys221/data/ABAW/cropped_aligned/' + x for x in au_video]
    len_images = []
    for i in au_video:
        a = glob.glob(i + '/*')
        len_images.append(len(a))

    df = pd.DataFrame(columns=['folder_dir'], data=au_video)
    df['len_images'] = len_images
    df['name_videos'] = df.folder_dir.apply(lambda x: x.replace('/data/users/ys221/data/ABAW/cropped_aligned/', ''))

    df3 = df.merge(df1, how='outer', on='name_videos')
    df3 = df3.rename(columns={'len_images_x': 'num_images', 'len_images_y': 'num_frames'})

    df3.to_csv(out_path)


def ex_count_weight(labels_ex):
    counts = []
    for i in range(8):
        labels = list(labels_ex)
        counts.append(labels.count(i))
    N = len(labels_ex)
    r = [N / counts[i] if counts[i] != 0 else 0 for i in range(8)]
    s = sum(r)
    weight = [r[i] / s for i in range(8)]

    return torch.as_tensor(weight, dtype=torch.float), counts


def ex_weight(labels_ex):
    counts = []
    for i in range(8):
        labels = list(labels_ex)
        counts.append(labels.count(i))
    N = len(labels_ex)
    r = [N / counts[i] if counts[i] != 0 else 0 for i in range(8)]
    s = sum(r)
    weight = [r[i] / s for i in range(8)]

    return torch.as_tensor(weight, dtype=torch.float)


def split_csv_data(file1, file2, number_fold):
    if file2 is not None:
        list_csv1 = glob.glob(file1)
        list_csv2 = glob.glob(file2)
        list_csv = list_csv1 + list_csv2
    else:
        list_csv = glob.glob(file1)

    kf = KFold(n_splits=number_fold)
    csv_train = {}
    csv_valid = {}

    for fold, (train_index, valid_index) in enumerate(kf.split(list_csv)):
        df_train = pd.DataFrame()
        for i in tqdm(train_index, total=len(train_index)):
            df_train = pd.concat((df_train, pd.read_csv(list_csv[i])), axis=0).reset_index(drop=True)

        df_valid = pd.DataFrame()
        for j in tqdm(valid_index, total=len(valid_index)):
            df_valid = pd.concat((df_valid, pd.read_csv(list_csv[j])), axis=0).reset_index(drop=True)

        csv_train[fold] = df_train
        csv_valid[fold] = df_valid

    return csv_train, csv_valid


def split_data(df_data1, df_data2, number_fold):
    if df_data2 is not None:
        df_data1 = pd.read_csv(df_data1, index_col=False)
        df_data2 = pd.read_csv(df_data2, index_col=False)
        data = pd.concat((df_data1, df_data2), axis=0)
    else:
        data = df_data1

    kf = GroupKFold(n_splits=number_fold)
    df_train = {}
    df_valid = {}
    for fold, (train_index, valid_index) in enumerate(kf.split(data, data, data.iloc[:, 0])):
        df_train[fold] = data.iloc[train_index].reset_index(drop=True)
        df_valid[fold] = data.iloc[valid_index].reset_index(drop=True)

    return df_train, df_valid


def Val_acc(loader, Dis, criterion, device):
    '''
    validation function based on self-built models
    :param loader: data loader
    :param Dis: discriminator object
    :param criterion: criterion to calculate the loss
    :return: accuracy and loss
    '''
    # predictions result
    pre_list = []
    # ground-truth
    GT_list = []
    val_ce = 0

    for i, (batch_val_x, batch_val_y) in enumerate(loader):
        GT_list = np.hstack((GT_list, batch_val_y.numpy()))
        batch_val_x = Variable(batch_val_x).to(device)
        batch_val_y = Variable(batch_val_y).to(device)
        # inference
        _, batch_p = Dis(batch_val_x)

        batch_result = batch_p.cpu().data.numpy().argmax(axis=1)
        pre_list = np.hstack((pre_list, batch_result))

        # classification loss
        val_ce += criterion(batch_p, batch_val_y).cpu().data.numpy()

    # calculate the accuracy
    val_acc = (np.sum((GT_list == pre_list).astype(float))) / len(GT_list)
    val_ce = val_ce / i

    return val_acc, val_ce


def combinefig_dualcon(FR_mat, ER_mat, Fake_mat, con_FR, con_ER, save_num=3):
    '''
    combine five images to one row, combine three row of images to one complete image
    :param FR_mat: face images
    :param ER_mat: expression images
    :param Fake_mat: fake images
    :param con_FR: consistent images with respect to FR
    :param con_ER: consistent images with respect to ER
    :return: combined images
    '''
    save_num = min(FR_mat.shape[0], save_num)
    imgsize = np.shape(FR_mat)[-1]
    img = np.zeros([imgsize * save_num, imgsize * 5, 3])
    for i in range(0, save_num):
        img[i * imgsize: (i + 1) * imgsize, 0 * imgsize: 1 * imgsize, :] = FR_mat[i, :, :, :].transpose([1, 2, 0])
        img[i * imgsize: (i + 1) * imgsize, 1 * imgsize: 2 * imgsize, :] = ER_mat[i, :, :, :].transpose([1, 2, 0])
        img[i * imgsize: (i + 1) * imgsize, 2 * imgsize: 3 * imgsize, :] = Fake_mat[i, :, :, :].transpose([1, 2, 0])
        img[i * imgsize: (i + 1) * imgsize, 3 * imgsize: 4 * imgsize, :] = con_FR[i, :, :, :].transpose([1, 2, 0])
        img[i * imgsize: (i + 1) * imgsize, 4 * imgsize: 5 * imgsize, :] = con_ER[i, :, :, :].transpose([1, 2, 0])

    return img


def Val_acc_single(x_ER, Dis, device, name):
    # the meaning of each class in RAF-DB
    exprdict = {
        0: 'Neutral',
        1: 'Anger',
        2: 'Disgust',
        3: 'Fear',
        4: 'Happiness',
        5: 'Sadness',
        6: 'Surprise',
        7: 'Other',
        # 0: 'Surprise',
        # 1: 'Fear',
        # 2: 'Disgust',
        # 3: 'Happiness',
        # 4: 'Sadness',
        # 5: 'Anger',
        # 6: 'Neutral',
    }
    x_ER = Variable(x_ER).to(device)
    # inference
    _, x_p = Dis(x_ER)
    pred_cls = x_p.cpu().data.numpy().argmax(axis=1).item()
    print('the predicted class of model {} is: {}'.format(name, exprdict[pred_cls]))


def del_extra_keys(model_par_dir):
    # the pretrained model is trained on old version pytorch, some extra keys should be deleted before loading
    model_par_dict = torch.load(model_par_dir)
    model_par_dict_clone = model_par_dict.copy()
    # delete keys
    for key, value in model_par_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_par_dict[key]
    return model_par_dict







