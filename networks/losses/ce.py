import torch
import torch.nn as nn

from utils.common import calc_weights

def weighted_binary_cross_entropy_loss(y_hat, y_true):
    # Calculate weights for targets
    weights = calc_weights(y_true)

    # Calculate binary cross entropy
    bce = nn.BCEWithLogitsLoss()(y_hat, y_true)

    # Apply the weights
    weight_vector = y_true * weights['structure'] + (1. - y_true) * weights['background']
    weighted_bce = weight_vector * bce

    return weighted_bce.mean()
