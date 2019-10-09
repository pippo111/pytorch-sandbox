# Models common setup
setup = {
    'dataset_dir': '/home/filip/Projekty/ML/datasets/processed/mindboggle_84_Nx192x256_lateral_ventricle',
    'struct': 'lateral_ventricle',
    'epochs': 2,
    'batch_size': 16,
    'seed': 5
}

# Model different parameters
models = [
    {
        'arch': 'Unet', 'filters': 16, 'lr': 1e-3,
        'loss_fn': 'weighted_bce'
    },
    {
        'arch': 'Unet', 'filters': 16, 'lr': 1e-3,
        'loss_fn': 'bce'
    }
]
