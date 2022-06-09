import torch
import torchvision
from torch import nn

from models.model_MLP import MLP


class RES3D_feature(nn.Module):
    def __init__(self, module='R3D', num_classes=8, drop_out=0.5, dim=400, mode='IMG'):
        super(RES3D_feature, self).__init__()
        if module=='R3D':
            self.pretrain = torchvision.models.video.r3d_18(pretrained=True, num_classes=dim).cuda()
        elif module=='MC3':
            self.pretrain = torchvision.models.video.mc3_18(pretrained=True, num_classes=dim).cuda()
        else:
            ValueError

        in_features = self.pretrain.fc.in_features

        # add mlp projection head
        # self.pretrain.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.pretrain.fc)
        self.pretrain = nn.Identity()
        self.out = MLP([in_features, in_features/2, num_classes], drop_out=drop_out)


    def forward(self, x):
        # if self.mode == 'IMG':
        #     x = torch.unsqueeze(x, dim=2)
        # elif self.mode == 'SCHW':
        #     x = x.permute(0, 2, 1, 3, 4)
        # else:
        #     pass
        # self.pretrain.eval()
        return self.out(self.pretrain(x))


