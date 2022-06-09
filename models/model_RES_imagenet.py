import torch
import torchvision
from torch import nn

from models.model_MLP import MLP


class RES_feature(nn.Module):
    def __init__(self, drop_out = 0.5, *args, ** kwargs):
        self.args = kwargs['args']
        module = self.args.arch
        mode = self.args.mode
        num_classes = self.args.num_classes

        super(RES_feature, self).__init__()
        if module=='resnet18':
            self.pretrain = torchvision.models.resnet18(pretrained=True).cuda()
        elif module=='resnet50':
            self.pretrain = torchvision.models.resnet50(pretrained=True).cuda()
        elif module=='resnet101':
            self.pretrain = torchvision.models.resnet101(pretrained=True).cuda()
        elif module=='resnet152':
            self.pretrain = torchvision.models.resnet152(pretrained=True).cuda()
        else:
            self.pretrain = torchvision.models.resnet34(pretrained=True).cuda()

        if mode == 'audio':
            self.pretrain.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        dim_mlp = self.pretrain.fc.in_features
        dim = dim_mlp
        self.pretrain.fc = nn.Identity()

        # dim = self.pretrain.fc.out_features

        # add mlp projection head
        # self.pretrain.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.pretrain.fc)

        self.out = MLP([dim, dim/2, dim/4, num_classes], drop_out=drop_out)

    def forward(self, x):
        # self.pretrain.eval()
        return self.out(self.pretrain(x))


