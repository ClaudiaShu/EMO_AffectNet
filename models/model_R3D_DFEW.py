import torch
import torchvision
from torch import nn

from models.model_MLP import MLP


class DFEW_SSL(nn.Module):
    def __init__(self, num_classes=8, drop_out=0.5, mode='IMG'):
        super(DFEW_SSL, self).__init__()
        self.pretrain = torchvision.models.video.resnet.mc3_18(pretrained=False, num_classes=num_classes).cuda()
        model_state_dict = '/data/users/ys221/data/models/model_voxceleb_v_simCLR/checkpoint_0085.pth.tar'
        checkpoint = torch.load(model_state_dict, map_location='cuda:0')
        state_dict = checkpoint['state_dict']

        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    # remove prefix
                    state_dict[k[len("backbone."):]] = state_dict[k]
            del state_dict[k]

        in_features = self.pretrain.fc.in_features
        self.pretrain.fc = nn.Identity()
        self.pretrain.load_state_dict(state_dict, strict=False)
        # self.fea = MLP([in_features, in_features], drop_out=drop_out)
        # self.fea.load_state_dict(out_dict, strict=False)

        self.out = MLP([in_features, in_features/2, num_classes], drop_out=drop_out)
        self.mode = mode

    def forward(self, x):
        # if self.mode == 'IMG':
        #     x = torch.unsqueeze(x, dim=2)
        # elif self.mode == 'SCHW':
        #     x = x.permute(0, 2, 1, 3, 4)
        # else:
        #     pass
        # self.pretrain.eval()
        return self.out(self.pretrain(x))


