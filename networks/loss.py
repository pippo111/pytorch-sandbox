import torch.nn as nn

from networks.losses.dice import dice_coef_loss
from networks.losses.dice import weighted_dice_coef_loss
from networks.losses.ce import weighted_binary_cross_entropy_loss

def get(name):
    loss_fn = dict(
        bce = nn.BCEWithLogitsLoss(),
        dice = dice_coef_loss,
        weighted_bce = weighted_binary_cross_entropy_loss,
        weighted_dice = weighted_dice_coef_loss
    )

    return loss_fn[name]
