import torch
import torchvision
from torch import nn

from models.model_MLP import MLP


# Recurrent
class BaselineRNN(nn.Module):
    def __init__(self, num_classes=8, module='R3D', n_features=512,
                 hidden_size=512, num_layers=2, drop_gru=0.4,
                 embed_dim=512, num_heads=2, drop_att=0.7):
        super(BaselineRNN, self).__init__()
        '''
        r3d_18 & mc3_18: 
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        N → number of sequences (mini batch)
        Cin → number of channels (3 for rgb)
        D → Number of images in a sequence
        H → Height of one image in the sequence
        W → Width of one image in the sequence
        '''
        self.baseline = torchvision.models.resnet18(pretrained=True).cuda()
        embed_dim = self.baseline.fc.in_features
        self.baseline.fc = nn.Identity()
        # pretrained on DEFW
        if module == "R3D":
            pretrainedr3d = torch.load(
                "/data/users/ys221/data/pretrain/resnet/r3d_18/r3d_18_fold3_epo144_UAR45.69_WAR56.92.pth")
            self.backbone = torchvision.models.video.resnet.r3d_18(pretrained=pretrainedr3d).cuda()
        elif module == "MC3":
            pretrainedmc3 = torch.load(
                "/data/users/ys221/data/pretrain/resnet/mc3_18/mc3_18_fold3_epo038_UAR46.85_WAR58.93.pth")
            self.backbone = torchvision.models.video.resnet.mc3_18(pretrained=pretrainedmc3).cuda()
        else:
            self.backbone = torchvision.models.video.resnet.r2plus1d_18(pretrained=True).cuda()

        self.backbone.fc = nn.Identity()  # 512

        # attention
        self.att = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=drop_att).cuda()

        # GRU
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, bidirectional=False, batch_first=True,
                          dropout=drop_gru, num_layers=num_layers).cuda()

        self.out = MLP([1024, 512, 256, num_classes], drop_out=0.5)  # class

    def forward(self, x):
        '''
        e input clip of size 3 × L × H × W, where L
        is the number of frames in the clip
        L, 3, H, W
        :param x: bz, seq, c, h, w
        :return: output
        '''
        # x_rnn = x.permute(0, 2, 1, 3, 4)
        out_rnn = self.backbone(x)

        x = x.permute(0, 2, 1, 3, 4)
        b, l, c, h, w = x.shape
        x_att = x.reshape(b * l, c, h, w)
        out_res = self.baseline(x_att)
        out_res = out_res.reshape(b, l, -1)
        out_att, _ = self.att(out_res, out_res, out_res)
        out_att = out_att.cuda()
        # out_gru, _ = self.gru(out_att)

        out_att = out_att[:, -1].cuda()
        # out_gru = out_gru[:, -1].cuda()

        out = torch.cat([out_rnn, out_att], dim=1)
        out = self.out(out)

        return out

