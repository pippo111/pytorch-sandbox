import torch

def dice_coef_loss(y_true, y_pred):
    def dice_coef(y_true, y_pred, smooth=1.0):
        return (2 * torch.sum(y_true * y_pred) + 1) / (torch.sum(y_true + y_pred) + 1)

    return 1 - dice_coef(y_true, y_pred)

def weighted_dice_coef_loss(y_true, y_pred, weights):
    pass