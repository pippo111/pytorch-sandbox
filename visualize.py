import numpy as np
from torch.utils.data import DataLoader

import config as cfg

from networks.model import MyModel
from utils.dataset import get_loader

# Model setup
model = {
    'arch': 'Unet', 'filters': 2,
    'filename': 'wmh_Unet_Adam_bce_bs-16_f-2.pt'
}

# Create train / validation loaders
dataset_dir = cfg.setup['dataset_dir']
test_loader = get_loader(dataset_dir, 'test', shuffle = False)

# For each model setup perform training
my_model = MyModel(
    arch = model['arch'],
    struct = cfg.setup['struct'],
    n_filters = model['filters']
)

my_model.load(model['filename'])
my_model.visualize(test_loader)