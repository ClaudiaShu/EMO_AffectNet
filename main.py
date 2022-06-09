import argparse
import warnings

import torch

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from ABAW_main import ABAW_trainer
from data_aug import get_image_transform, get_audio_transform
from data_loader import AFFECTNET_Dataset, Aff2_Dataset_static_shuffle, Aff2_Dataset_audio
from loss.balanced_softmax_cross_entropy_loss import BalancedSoftmaxCE
from loss.class_balanced_loss import ClassBalanceCE
from models.model_R3D_DFEW import DFEW_SSL
from models.model_RES import RES_SSL
from models.model_RES_imagenet import RES_feature
from utils import seed_everything, create_original_data, FocalLoss
from utils.prefetch_dataloader import DataLoaderX

from config import *

warnings.filterwarnings("ignore")
device = "cuda:0" if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='EXP Training')
parser.add_argument('--comment', type=str, default='')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')  # 1e-4#
parser.add_argument('--batch_size', default=64, type=int, help='batch size')  # 64#
parser.add_argument('--epochs', default=20, type=int, help='number epochs')  # 20#
parser.add_argument('--num_classes', default=8, type=int, help='number classes')
parser.add_argument('-weight_decay', default=5e-4, type=float)  # 5e-4#
parser.add_argument('--seq_len', default=3, type=int)
parser.add_argument('--sec', default=1)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--resume', type=bool, default=False)

# parser.add_argument('--loss', type=str, default='CrossEntropyLoss')  # ['CrossEntropyLabelAwareSmooth','SuperContrastive']
# parser.add_argument('--warmup', type=bool, default=False)
# parser.add_argument('--optim', type=str, default='ADAM')

parser.add_argument('--arch', default='resnet50', type=str, help='baseline of the training network.')
parser.add_argument('--rep', default='pretrain', type=str, help='Choose methods for representation learning.') # ['SSL','pretrain']
parser.add_argument('--dataset', default='AffectNet') # AffectNet Aff2
parser.add_argument('--mode', default='image') # image audio
parser.add_argument('--model_dir', default='/mnt/d/Data/Yuxuan/logging/simCLR/VoxCeleb_img/runs/resnet50_voxceleb1s_img/checkpoint_0090.pth.tar')
parser.add_argument('--state_dict', default="/mnt/d/Data/Yuxuan/downstream/image/runs/Jun07_11-20-10_ic_xiao/checkpoint_bast.pth.tar")

parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')

args = parser.parse_args()

def setup_Aff2(args):
    # train set
    df_train = create_original_data(
        f'/mnt/c/Data/Yuxuan/ABAW/labels_save/expression/Train_Set_v3/*')
    # valid set
    df_valid = create_original_data(
        f'/mnt/c/Data/Yuxuan/ABAW/labels_save/expression/Validation_Set_v3/*')

    if args.mode == 'image':
        train_transform, test_transform = get_image_transform()
        train_dataset = Aff2_Dataset_static_shuffle(df=df_train, root=False, transform=train_transform)
        valid_dataset = Aff2_Dataset_static_shuffle(df=df_valid, root=False, transform=test_transform)
    elif args.mode == 'audio':
        train_transform_aud, test_transform_aud = get_audio_transform()
        train_dataset = Aff2_Dataset_audio(df=df_train, sec=args.sec, root=False, transform_aud=train_transform_aud)
        valid_dataset = Aff2_Dataset_audio(df=df_valid, sec=args.sec, root=False, transform_aud=test_transform_aud)
    else:
        raise ValueError('invalid mode')

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

def setup_AffectNet(args):
    train_transform, valid_transform = get_image_transform()
    # train set
    train_data = "/mnt/c/Data/Yuxuan/AffectNet/train_set_data.csv"
    valid_data = "/mnt/c/Data/Yuxuan/AffectNet/val_set_data.csv"

    train_dataset = AFFECTNET_Dataset(df=train_data,
                                root=False,
                                transform=train_transform)
    valid_dataset = AFFECTNET_Dataset(df=valid_data,
                                      root=False,
                                      transform=train_transform)

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

    if args.dataset == 'Aff2':
        train_loader, valid_loader = setup_Aff2(args)
    elif args.dataset == 'AffectNet':
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

    num_class_list = [74874, 134415, 25459, 14090, 6378, 3803, 24882, 3750]
    para_dict = {
        "num_classes": args.num_classes,
        "num_class_list": num_class_list,
        "device": device,
        "cfg": cfg_loss,
    }
    # criterion = nn.CrossEntropyLoss().to(args.device)
    # criterion = FocalLoss(args.num_classes, batch_size=args.batch_size).to(args.device)
    criterion = ClassBalanceCE(para_dict).to(args.device)
    # criterion = BalancedSoftmaxCE(para_dict).to(args.device)

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
        exp_train = ABAW_trainer(best_acc,
                                 model=model,
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 criterion=criterion,
                                 args=args)
        exp_train.run(train_loader, valid_loader)


if __name__ == "__main__":
    main()