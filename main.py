import argparse
import warnings

import torch

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from ABAW_main import ABAW_trainer
from data_aug import get_cv_image_transform
from data_loader import AFFECTNET_Dataset, AFFECTNET_hdf5
from loss.balanced_softmax_cross_entropy_loss import BalancedSoftmaxCE
from loss.class_balanced_loss import ClassBalanceCE
from models.model_FaceCycle import Cycle_feature

from models.model_RES import RES_SSL
from models.model_RES_imagenet import RES_feature
from utils import seed_everything, create_original_data, FocalLoss, CCCLoss

from config import *

warnings.filterwarnings("ignore")
device = "cuda:0" if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='EXP Training')
parser.add_argument('--comment', type=str, default='')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')  # 1e-4#
parser.add_argument('--batch_size', default=64, type=int, help='batch size')  # 64#
parser.add_argument('--epochs', default=20, type=int, help='number epochs')  # 20#
# parser.add_argument('--num_classes', default=8, type=int, help='number classes')
parser.add_argument('-weight_decay', default=5e-4, type=float)  # 5e-4#
parser.add_argument('--seq_len', default=3, type=int)
parser.add_argument('--sec', default=1)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--resume', type=bool, default=False)

# parser.add_argument('--loss', type=str, default='CrossEntropyLoss')  # ['CrossEntropyLabelAwareSmooth','SuperContrastive']
# parser.add_argumenat('--warmup', type=bool, default=False)
# parser.add_argument('--optim', type=str, default='ADAM')

parser.add_argument('--task', default='exp', type=str, choices=['exp','va'])
parser.add_argument('--arch', default='resnet50', type=str, help='baseline of the training network.')
parser.add_argument('--rep', default='SSL', type=str, help='Choose methods for representation learning.') # ['SSL','pretrain']
parser.add_argument('--dataset', default='AffectNet') # AffectNet Aff2
parser.add_argument('--mode', default='image') # image audio
parser.add_argument('--model_dir', default="/data2/yshu21/data/model/SSL_model/simclr_with_time_neg_img_c/checkpoint_best.pth.tar")
parser.add_argument('--state_dict', default="/mnt/d/Data/Yuxuan/downstream/image/runs/Jun07_11-20-10_ic_xiao/checkpoint_bast.pth.tar")
parser.add_argument('--loadmodel', default='/data2/yshu21/data/FaceCycle/FaceCycleModel.tar')

parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')

args = parser.parse_args()

def setup_AffectNet(args):
    train_transform, valid_transform = get_cv_image_transform(size=64)
    train_data = "/data2/ys221/data/AffectNet/train_data.csv"
    valid_data = "/data2/ys221/data/AffectNet/val_data.csv"

    train_dataset = AFFECTNET_Dataset(df=train_data,
                                task=args.task,
                                transform=train_transform)
    valid_dataset = AFFECTNET_Dataset(df=valid_data,
                                      task=args.task,
                                      transform=train_transform)
    # train_data = "/data2/yshu21/data/AffectNet/cache/AffectNet_train.hdf5"
    # valid_data = "/data2/yshu21/data/AffectNet/cache/AffectNet_valid.hdf5"
    #
    # train_dataset = AFFECTNET_hdf5(train_data, transform=train_transform, task=args.task)
    # valid_dataset = AFFECTNET_hdf5(valid_data, transform=valid_transform, task=args.task)
    #
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=False,
                              shuffle=True,
                              drop_last=False)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=False,
                              shuffle=False,
                              drop_last=False)

    return train_loader, valid_loader


def main():
    # args = parser.parse_args()
    # seed_everything()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')

    if args.dataset == 'AffectNet':
        train_loader, valid_loader = setup_AffectNet(args)
    else:
        raise ValueError('Invalid input of dataset')

    if args.rep == 'pretrain':
        model = RES_feature(args=args)
    elif args.rep == 'SSL':
        model = RES_SSL(args=args)
    else:
        raise ValueError('Invalid input of training model')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    num_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0,
                                                           last_epoch=-1)

    if args.task == 'exp':
        num_class_list = [74874, 134415, 25459, 14090, 6378, 3803, 24882, 3750]
        para_dict = {
            "num_classes": 8,
            "num_class_list": num_class_list,
            "device": device,
            "cfg": cfg_loss,
        }
        # criterion = nn.CrossEntropyLoss().to(args.device)
        # criterion = FocalLoss(args.num_classes, batch_size=args.batch_size).to(args.device)
        # criterion = ClassBalanceCE(para_dict).to(args.device)
        criterion = BalancedSoftmaxCE(para_dict).to(args.device)
    elif args.task == 'va':
        # criterion = nn.MSELoss()
        criterion = CCCLoss()
    else:
        raise ValueError('Invalid task')

    optimizer.zero_grad()
    optimizer.step()

    # resume training
    best_acc = 0
    if args.resume:
        # epoch
        model_state_dict = args.state_dict

        checkpoint = torch.load(model_state_dict, map_location='cuda:0')
        state_dict = checkpoint['state_dict']

        for k in list(state_dict.keys()):
            if k.startswith('pretrain.'):
                if k.startswith('pretrain') and not k.startswith('pretrain.fc'):
                    # remove prefix
                    state_dict[k[len("pretrain."):]] = state_dict[k]
            del state_dict[k]

        model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_acc = checkpoint['f1']

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        train = ABAW_trainer(best_acc,
                                 model=model,
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 criterion=criterion,
                                 args=args)
        train.run(train_loader, valid_loader)


if __name__ == "__main__":
    main()
