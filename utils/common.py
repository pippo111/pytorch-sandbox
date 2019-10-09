import os
import random
import numpy as np
import torch

def seed_torch(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def calc_weights(mask):
    class_share = { 'background': 0, 'structure': 0 }

    mask = mask.cpu().numpy()
    y = np.array(mask).argmax(axis=-1)
    nonzeros = np.count_nonzero(y)
    class_share['background'] = (y.size - nonzeros) / y.size
    class_share['structure'] = nonzeros / y.size

    return class_share
