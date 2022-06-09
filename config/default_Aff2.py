from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from yacs.config import CfgNode as CN


_C = CN()

# ----- HP BUILDER -----
_C.HP = CN()
_C.HP.LR = 1e-4
_C.HP.BATCH_SIZE = 64
_C.HP.EPOCHS = 20
_C.HP.NUM_CLASSES = 8
_C.HP.WEIGHT_DECAY = 5e-4
