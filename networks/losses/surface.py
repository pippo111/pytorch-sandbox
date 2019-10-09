import numpy as np
import torch

from utils.image import calc_dist_map

def surface_loss(y_hat, y_true):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # calculate distance and pass it to surface loss
    y_true_numpy = y_true.cpu().numpy()
    y_dist = np.array([calc_dist_map(y) for y in y_true_numpy]).astype(np.float32)
    y_dist = torch.from_numpy(y_dist).to(device)

    # calculate loss
    multipled = y_hat * y_dist
    loss = multipled.mean()

    return loss
