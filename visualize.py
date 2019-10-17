import numpy as np
from torch.utils.data import DataLoader

import config as cfg

from networks.model import MyModel
from utils.dataset import get_loader

# Model setup
model = {
    'arch': 'Unet', 'filters': 16,
    'filename': 'brain_Unet_Adam_bce_bs-16_f-16.pt'
}

# Create train / validation loaders
dataset_dir = '/home/filip/Projekty/ML/datasets/processed/wmh_all_80_192x256x256_wmh'
test_loader = get_loader(dataset_dir, 'test', shuffle = False)

# For each model setup perform training
my_model = MyModel(
    arch = model['arch'],
    struct = cfg.setup['struct'],
    n_filters = model['filters'],
    batch_size = cfg.setup['batch_size']
)

my_model.load(model['filename'])
my_model.visualize(test_loader)