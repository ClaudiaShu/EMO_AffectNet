import torch
import torchvision
from torch import nn

# model = torchvision.models.resnet18(pretrained=False, num_classes=10).cuda()
# model_state_dict = '/data/users/ys221/data/models/model_voxceleb_img_simCLR/May10_21-27-03_ic_xiao/checkpoint_0090.pth.tar'
# checkpoint = torch.load(model_state_dict, map_location='cuda:0')
# state_dict = checkpoint['state_dict']
#
# for k in list(state_dict.keys()):
#
#   if k.startswith('backbone.'):
#     if k.startswith('backbone') and not k.startswith('backbone.fc'):
#       # remove prefix
#       state_dict[k[len("backbone."):]] = state_dict[k]
#   del state_dict[k]
#
# dim_mlp = model.fc.in_features
# model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
#
# log = model.load_state_dict(state_dict, strict=False)

model = torchvision.models.video.mc3_18(pretrained=False)
print(model)