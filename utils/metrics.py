import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score, accuracy_score, average_precision_score, precision_recall_curve, confusion_matrix
from torch.autograd import Variable
from torch.nn import MSELoss, CrossEntropyLoss, L1Loss, SmoothL1Loss
import sklearn

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)

def get_WAR(trues_te, pres_te):
    WAR  = accuracy_score(trues_te, pres_te)
    return WAR

def get_UAR(trues_te, pres_te):
    cm = confusion_matrix(trues_te, pres_te)
    acc_per_cls = [ cm[i,i]/sum(cm[i]) for i in range(len(cm))]
    UAR = sum(acc_per_cls)/len(acc_per_cls)
    return UAR

def get_cm(trues_te, pres_te):
    cm = confusion_matrix(trues_te, pres_te)
    return cm

def adjust_learning_rate(epoch):
    lr = 1e-2
    if epoch >= 10 and epoch < 20:
        lr = 1e-3
    elif epoch >= 20 and epoch < 30:
        lr = 5e-3
    elif epoch >= 30 and epoch < 40:
        lr = 1e-4
    elif epoch >= 40:
        lr = 5e-4
    elif epoch == 0:
        lr = 1e-5
    return lr

def averaged_f1_score(y_pred, y_true):
    N, label_size = y_pred.shape
    f1s = []
    for i in range(label_size):
        f1 = f1_score(y_true[:, i], y_pred[:, i])
        f1s.append(f1)
    return np.mean(f1s), f1s


def accuracy(y_pred, y_true):
    assert len(y_pred.shape) == 1
    return sum(y_pred == y_true) / y_pred.shape[0]

def accuracy_top(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def averaged_accuracy(x, y):
    assert len(x.shape) == 2
    N, C = x.shape
    accs = []
    for i in range(C):
        acc = accuracy(x[:, i], y[:, i])
        accs.append(acc)
    return np.mean(accs), accs


def CCC_score(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2 * rho * x_s * y_s / (x_s ** 2 + y_s ** 2 + (x_m - y_m) ** 2)
    return ccc


class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, x, y):
        # the target y is continuous value (BS, )
        # the input x is either continuous value (BS, ) or probability output(digitized)
        y = y.view(-1)
        x = x.view(-1)
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))))
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2 * rho * x_s * y_s / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
        return 1 - ccc

class CCC_SmoothL1(nn.Module):
    def __init__(self):
        super(CCC_SmoothL1, self).__init__()

    def forward(self, x, y):
        loss1 = SmoothL1Loss()(x, y)
        loss2 = CCCLoss(x, y)
        return loss1 + loss2


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)

def info_nce_loss(self, features):
    labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(self.args.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

    logits = logits / self.args.temperature
    return logits, labels

def VA_metric(x, y):
    items = [CCC_score(x[:, 0], y[:, 0]), CCC_score(x[:, 1], y[:, 1])]
    return items, np.mean(items)


def EXPR_metric(y_pred, y_true):
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    return f1, acc, 0.67 * f1 + 0.33 * acc


def AU_metric(y_pred, y_true):
    f1, _ = averaged_f1_score(y_pred, y_true)
    acc, _ = averaged_accuracy(y_pred, y_true)
    return f1, acc, 0.5 * f1 + 0.5 * acc


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce
#
#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
#
#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss

class FocalLoss(nn.Module):
    """
    This is an implementation of Focal Loss
    Loss(x, class) = -(alpha (1-sigmoid(p)[class])^gamma * log(sigmoid(x)[class]) )
    Args:
        alpha (1D Tensor, Variable): the scalar factor for each class, similar to the weights in BCEWithLogitsLoss.weights
        gamma(float, double): gamma>0, reduces the loss for well-classified samples, putting more focus on hard, misclassified samples
        size_average (bool): by default, the losses are averaged over observations.
    """

    def __init__(self, class_num, batch_size, gamma=2, alpha=None, size_average=True, pos_weight=None,
                 activation='sigmoid'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.activation = activation
        if alpha is None:
            if pos_weight is None:
                self.alpha = Variable(torch.Tensor([1]))
            else:
                # alpha is determined by pos_weight
                self.alpha = Variable(pos_weight.data.clone())
        else:
            self.alpha = alpha
        self.class_num = class_num
        self.size_average = size_average
        self.label = torch.zeros((batch_size, class_num)).cuda()

    def forward(self, inputs, targets):
        # target size is (N,) if task is EXPR; target size is (N,C) if task is AU;
        N, C = inputs.size()
        if self.activation == 'sigmoid':
            P = torch.sigmoid(inputs)
        elif self.activation == 'softmax':
            P = F.softmax(inputs, dim=-1)
        if not len(targets.size()) == len(inputs.size()):
            assert len(targets.size()) == 1 or (targets.size()[1] == 1)
            self.label.resize_((N, C)).copy_(F.one_hot(targets.view(-1).data, C))
            targets = Variable(self.label)
        pt_1 = torch.where(targets == 1, P, torch.ones_like(P))
        pt_0 = torch.where(targets == 0, P, torch.zeros_like(P))
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        if inputs.is_cuda and not pt_0.is_cuda:
            pt_0 = pt_0.cuda()
        if inputs.is_cuda and not pt_1.is_cuda:
            pt_1 = pt_1.cuda()
        loss = - (self.alpha * torch.pow(1. - pt_1, self.gamma) * pt_1.log() + torch.pow(pt_0, self.gamma) * (
                1. - pt_0).log()).sum(dim=1)
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def test_scikit_ap(cat_preds, cat_labels, ind2cat):
    ''' Calculate average precision per emotion category using sklearn library.
    :param cat_preds: Categorical emotion predictions.
    :param cat_labels: Categorical emotion labels.
    :param ind2cat: Dictionary converting integer index to categorical emotion.
    :return: Numpy array containing average precision per emotion category.
    '''
    ap = np.zeros(26, dtype=np.float32)
    for i in range(26):
        ap[i] = average_precision_score(cat_labels[i, :], cat_preds[i, :])
        print('Category %16s %.5f' % (ind2cat[i], ap[i]))
    print('Mean AP %.5f' % (ap.mean()))
    return ap


def get_thresholds(cat_preds, cat_labels):
    ''' Calculate thresholds where precision is equal to recall. These thresholds are then later for inference.
    :param cat_preds: Categorical emotion predictions.
    :param cat_labels: Categorical emotion labels.
    :return: Numpy array containing thresholds per emotion category where precision is equal to recall.
    '''
    thresholds = np.zeros(12, dtype=np.float32)
    for i in range(12):
        p, r, t = precision_recall_curve(cat_labels[i, :], cat_preds[i, :])
        for k in range(len(p)):
            if p[k] == r[k]:
                thresholds[i] = t[k]
                break
    return thresholds


class FocalLoss_Ori(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor
        gamma:
        ignore_index:
        reduction:
    """

    def __init__(self, num_class, alpha=None, gamma=2, ignore_index=None, reduction='mean'):
        super(FocalLoss_Ori, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.ignore_index = ignore_index
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, )
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

        # if isinstance(self.alpha, (list, tuple, np.ndarray)):
        #     assert len(self.alpha) == self.num_class
        #     self.alpha = torch.Tensor(list(self.alpha))
        # elif isinstance(self.alpha, (float, int)):
        #     assert 0 < self.alpha < 1.0, 'alpha should be in `(0,1)`)'
        #     assert balance_index > -1
        #     alpha = torch.ones((self.num_class))
        #     alpha *= 1 - self.alpha
        #     alpha[balance_index] = self.alpha
        #     self.alpha = alpha
        # elif isinstance(self.alpha, torch.Tensor):
        #     self.alpha = self.alpha
        # else:
        #     raise TypeError('Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        N, C = logit.shape[:2]
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)
        if prob.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        ori_shp = target.shape
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]
        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target * valid_mask

        # ----------memory saving way--------
        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.view(-1))
        alpha_class = alpha[target.squeeze().long()]
        class_weight = -alpha_class * torch.pow(torch.sub(1.0, prob), self.gamma)
        loss = class_weight * logpt
        if valid_mask is not None:
            loss = loss * valid_mask.squeeze()

        if self.reduction == 'mean':
            loss = loss.mean()
            if valid_mask is not None:
                loss = loss.sum() / valid_mask.sum()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)
        return loss