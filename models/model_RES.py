import torch
import torchvision
from torch import nn

from models.model_MLP import MLP

class resnet50(nn.Module):
    def __init__(self, dim=128, drop_out=0.5, *args, **kwargs):
        super(resnet50, self).__init__()
        self.args = kwargs['args']
        mode = self.args.mode
        num_classes = self.args.num_classes
        rep = self.args.rep
        if rep == 'SSL':
            dir = self.args.model_dir
            self.pretrain = torchvision.models.resnet50(pretrained=False).cuda()
            model_state_dict = dir

            checkpoint = torch.load(model_state_dict, map_location='cuda:0')
            state_dict = checkpoint['state_dict']

            for k in list(state_dict.keys()):

                if k.startswith('backbone.'):
                    if k.startswith('backbone') and not k.startswith('backbone.fc'):
                        # remove prefix
                        state_dict[k[len("backbone."):]] = state_dict[k]
                del state_dict[k]

            dim = self.pretrain.fc.in_features
            self.pretrain.fc = nn.Identity()

            self.pretrain.load_state_dict(state_dict, strict=False)
        elif rep == 'pretrain':
            dim = self.pretrain.fc.in_features
            self.pretrain.fc = nn.Identity()
            self.pretrain = torchvision.models.resnet50(pretrained=True).cuda()
        else:
            raise ValueError
        self.out = MLP([dim, dim / 2, dim / 4, num_classes], drop_out=drop_out)
    def forward(self, x):
        # self.pretrain.eval()
        return self.out(self.pretrain(x))

class RES_SSL(nn.Module):
    def __init__(self, dim=128, drop_out=0.5, *args, **kwargs):
        super(RES_SSL, self).__init__()
        self.args = kwargs['args']
        module = self.args.arch
        mode = self.args.mode
        num_classes = self.args.num_classes
        dir = self.args.model_dir

        if module=='resnet18':
            self.pretrain = torchvision.models.resnet18(pretrained=False, num_classes=dim).cuda()
            model_state_dict = '/data/users/ys221/data/models/model_voxceleb_img_simCLR/May10_21-27-03_ic_xiao/checkpoint_0090.pth.tar'
        elif module=='resnet50':
            self.pretrain = torchvision.models.resnet50(pretrained=False, num_classes=dim).cuda()
            model_state_dict = dir
        elif module=='resnet101':
            self.pretrain = torchvision.models.resnet101(pretrained=False, num_classes=dim).cuda()
            model_state_dict = '/data/users/ys221/data/models/model_voxceleb_img_simCLR/May17_00-21-33_ic_xiao/checkpoint_0010.pth.tar'
        else:
            self.pretrain = torchvision.models.resnet152(pretrained=False, num_classes=dim).cuda()
            model_state_dict = ''

        if mode == 'audio':
            self.pretrain.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        checkpoint = torch.load(model_state_dict, map_location='cuda:0')
        state_dict = checkpoint['state_dict']

        for k in list(state_dict.keys()):

            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    # remove prefix
                    state_dict[k[len("backbone."):]] = state_dict[k]
            del state_dict[k]

        dim = self.pretrain.fc.in_features
        self.pretrain.fc = nn.Identity()

        # add mlp projection head
        # self.pretrain.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.pretrain.fc)

        self.pretrain.load_state_dict(state_dict, strict=False)
        self.out = MLP([dim, dim/2, dim/4, num_classes], drop_out=drop_out)

    def forward(self, x):
        # self.pretrain.eval()
        return self.out(self.pretrain(x))


