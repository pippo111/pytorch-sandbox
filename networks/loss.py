import torch.nn as nn

from networks.losses.dice import dice_coef_loss
from networks.losses.ce import weighted_binary_cross_entropy_loss
from networks.losses.surface import surface_loss

def get(name):
    loss_fn = dict(
        bce = nn.BCEWithLogitsLoss(),
        weighted_bce = weighted_binary_cross_entropy_loss,
        dice = dice_coef_loss,
        surface = surface_loss
    )

    return loss_fn[name]
