import torch.nn as nn

from networks.losses.dice import dice_coef_loss
from networks.losses.gdl import GeneralizedDiceLoss
from networks.losses.ce import weighted_binary_cross_entropy_loss
from networks.losses.surface import surface_loss
from networks.losses.boundary import boundary_gdl_loss
from networks.losses.boundary import boundary_dice_loss
from networks.losses.boundary import boundary_bce_loss
from networks.losses.boundary import boundary_wbce_loss

def get(name):
    loss_fn = dict(
        bce = nn.BCEWithLogitsLoss(),
        weighted_bce = weighted_binary_cross_entropy_loss,
        dice = dice_coef_loss,
        gdl = GeneralizedDiceLoss(),
        surface = surface_loss,
        boundary_gdl = boundary_gdl_loss,
        boundary_dice = boundary_dice_loss,
        boundary_bce = boundary_bce_loss,
        boundary_wbce = boundary_wbce_loss
    )

    return loss_fn[name]
