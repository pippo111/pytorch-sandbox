import torch

def dice_coef_loss(y_hat, y_true):
    def dice_coef(y_hat, y_true, smooth=1.0):
        return (2 * torch.sum(y_true * y_hat) + 1) / (torch.sum(y_true + y_hat) + 1)

    return 1 - dice_coef(y_hat, y_true)
