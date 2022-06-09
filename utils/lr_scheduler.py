from bisect import bisect_right
import torch
import matplotlib.pyplot as plt
import numpy as np

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,  # [40,70]
            gamma=0.1,  #
            warmup_factor=0.01,
            warmup_iters=10,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):  # 保证输入的list是按前后顺序放的
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted",
                " but got {}".format(warmup_method)
            )

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    '''
    self.last_epoch是一直变动的[0,1,2,3,,,50]
    self.warmup_iters=10固定（表示线性warm up提升10个epoch）

    '''

    def get_lr(self):
        warmup_factor = 1
        list = {}
        if self.last_epoch < self.warmup_iters:  # 0<10
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor  # 1/3
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters  # self.last_epoch是一直变动的[0,1,2,3,,,50]/10
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha  # self.warmup_factor=1/3
                list = {"last_epoch": self.last_epoch, "warmup_iters": self.warmup_iters, "alpha": alpha,
                        'warmup_factor': warmup_factor}

        # print(base_lr  for base_lr in    self.base_lrs)
        # print(base_lr* warmup_factor* self.gamma ** bisect_right(self.milestones, self.last_epoch) for base_lr in self.base_lrs)

        return [base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch) for base_lr in
                self.base_lrs]  # self.base_lrs,optimizer初始学习率weight_lr=0.0003，bias_lr=0.0006

