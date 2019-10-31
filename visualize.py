import numpy as np
from torch.utils.data import DataLoader

import config as cfg

from networks.model import MyModel
from utils.dataset import get_loader

# Model setup
model = {
    'arch': 'Unet', 'filters': 16,
    'filename': 'cerebellum_wm_Unet_RAdam_boundary_dice_bs-16_f-16.pt'
}

# Create train / validation loaders
dataset_dir = '/home/filip/Projekty/ML/datasets/processed/mindboggle_84_Nx192x256_cerebellum_wm'
test_loader = get_loader(dataset_dir, 'test', shuffle = False)

# Get brainer
my_brainer = MyModel(
    arch = 'Unet',
    struct = 'brain',
    n_filters = 16,
    batch_size = 16
)

my_brainer.load('brain_Unet_RAdam_bce_bs-16_f-16.pt')

# For each model setup perform training
my_model = MyModel(
    arch = model['arch'],
    struct = cfg.setup['struct'],
    n_filters = model['filters'],
    batch_size = cfg.setup['batch_size'],
    brainer = my_brainer
)

my_model.load(model['filename'])
my_model.visualize(test_loader, saveAs=cfg.setup['struct'])