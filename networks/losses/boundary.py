import torch.nn as nn

from networks.losses.gdl import GeneralizedDiceLoss
from networks.losses.dice import dice_coef_loss
from networks.losses.ce import weighted_binary_cross_entropy_loss
from networks.losses.surface import surface_loss

def boundary_gdl_loss(y_hat, y_true, alpha):
    gloss = GeneralizedDiceLoss()(y_hat, y_true)
    sloss = surface_loss(y_hat, y_true)

    loss = alpha * gloss + (1 - alpha) * sloss

    return loss

def boundary_dice_loss(y_hat, y_true, alpha):
    dloss = dice_coef_loss(y_hat, y_true)
    sloss = surface_loss(y_hat, y_true)

    loss = alpha * dloss + (1 - alpha) * sloss

    return loss

def boundary_wbce_loss(y_hat, y_true, alpha):
    bloss = weighted_binary_cross_entropy_loss(y_hat, y_true)
    sloss = surface_loss(y_hat, y_true)

    loss = alpha * bloss + (1 - alpha) * sloss

    return loss

def boundary_bce_loss(y_hat, y_true, alpha):
    bloss = bce = nn.BCEWithLogitsLoss()(y_hat, y_true)
    sloss = surface_loss(y_hat, y_true)

    loss = alpha * bloss + (1 - alpha) * sloss

    return loss
