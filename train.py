import numpy as np
from torch.utils.data import DataLoader

import config as cfg

from networks.model import MyModel
from utils.dataset import get_loader
from utils.common import seed_torch

# Set the seed for deterministic results
seed_torch(10)

# Create train / validation loaders
dataset_dir = cfg.setup['dataset_dir']
train_loader = get_loader(dataset_dir, 'train', shuffle = True)
valid_loader = get_loader(dataset_dir, 'valid', shuffle = False)

# Get stripper
# For each model setup perform training
my_stripper = MyModel(
    arch = 'Unet',
    struct = 'brain',
    n_filters = 16,
    batch_size = 16
)

my_stripper.load('brain_Unet_Adam_bce_bs-16_f-16.pt')

# For each model setup perform training
for model in cfg.models:
    my_model = MyModel(
        arch = model['arch'],
        struct = cfg.setup['struct'],
        n_filters = model['filters'],
        batch_size = cfg.setup['batch_size']
    )

    history = my_model.train(
        epochs = cfg.setup['epochs'],
        train_loader = train_loader,
        valid_loader = valid_loader,
        loss_name = model['loss_fn'],
        optimizer_name = model['optimizer_fn'],
        stripper = my_stripper,
        learning_rate = model['lr']
    )

    my_model.save_results()
