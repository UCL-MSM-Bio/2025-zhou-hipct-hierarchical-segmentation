import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io as skio
import numpy as np
import config

def get_dice(pred, organ_target):
    '''
    Tversky loss
    :param pred: (B, C, 32, 256, 256)
    :param organ_target: (B, C, 32, 256, 256)
    '''
    # the organ_target should be one-hot code
    assert len(pred.shape) == len(organ_target.shape), 'the organ_target should be one-hot code'
    dice = 0
    for organ_index in range(1, config.NUM_CLASSES):
        P = pred[:, organ_index, :, :, :]
        _P = 1 - pred[:, organ_index, :, :, :]
        G = organ_target[:, organ_index, :, :, :]
        _G = 1 - organ_target[:, organ_index, :, :, :]
        mulPG = (P * G).sum(dim=1).sum(dim=1).sum(dim=1) # TP
        mul_PG = (_P * G).sum(dim=1).sum(dim=1).sum(dim=1) # FN
        mulP_G = (P * _G).sum(dim=1).sum(dim=1).sum(dim=1) # FP

        alpha = 0.7
        dice += (mulPG + 1) / (mulPG + alpha * mul_PG + (1-alpha) * mulP_G + 1)
        #dice += (2*mulPG) / (2*mulPG + mul_PG + mulP_G + 1e5)
        #print('dice: ', dice)
    return dice

# def get_dice(pred, organ_target):
#     '''
#     :param pred: (B, C, 32, 256, 256)
#     :param organ_target: (B, C, 32, 256, 256)
#     '''
#     # input and target shapes must match
#     assert pred.size() == organ_target.size(), "'pred' and 'target' must have the same shape"
#     dice = 0
#     for organ_index in range(1, config.NUM_CLASSES):
#         intersection = torch.sum(pred[:, organ_index, :, :, :] * organ_target[:, organ_index, :, :, :])
#         denominator = torch.sum(pred[:, organ_index, :, :, :]**2) + torch.sum(organ_target[:, organ_index, :, :, :]**2)
#         dice += (2. * intersection + 1) / (denominator + 1)
#     return dice

class DiceLoss(nn.Module):
    def __init__(self, apply_softmax=False):
        super().__init__()
        self.apply_softmax = apply_softmax

    def forward(self, pred, target):
        '''
        :param pred: (B, C, 32, 256, 256)
        :param target: (B, 32, 256, 256)
        '''
        if self.apply_softmax:
            pred = F.softmax(pred, dim=1)
        shape = target.shape
        organ_target = torch.zeros((target.size(0), config.NUM_CLASSES, shape[-3], shape[-2], shape[-1]))

        for organ_index in range(config.NUM_CLASSES):
            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1 
            # organ_index = 0 is background
            organ_target[:, organ_index, :, :, :] = temp_target

        organ_target = organ_target.cuda()

        return 1-get_dice(pred, organ_target).mean()

class DiceLoss_Focal(nn.Module):
    def __init__(self, apply_softmax=False):
        self.apply_softmax = apply_softmax
        super().__init__()

    def forward(self, pred, target):
        if self.apply_softmax:
            pred = F.softmax(pred, dim=1)
        shape = target.shape
        organ_target = torch.zeros((target.size(0), config.NUM_CLASSES, shape[-3], shape[-2], shape[-1]))

        for organ_index in range(config.NUM_CLASSES):
            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1
            organ_target[:, organ_index, :, :, :] = temp_target
            # organ_target: (B, 14, 48, 128, 128)

        organ_target = organ_target.cuda()

        pt_1 = get_dice(pred, organ_target).mean()
        gamma = 0.75
        return torch.pow((2-pt_1), gamma)


