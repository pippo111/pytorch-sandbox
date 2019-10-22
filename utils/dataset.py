import glob
import os

import torch
from torch.utils.data import Dataset, DataLoader

from utils.image import img_to_array, load_img

class Dataset2d(Dataset):
    def __init__(self, X_files, y_files):
        self.X = X_files
        self.y = y_files
        
    def __getitem__(self, index):
        X_image = img_to_array(load_img(self.X[index]))
        y_image = img_to_array(load_img(self.y[index]))

        X_tensor = torch.from_numpy(X_image)
        y_tensor = torch.from_numpy(y_image)

        return (X_tensor, y_tensor)
    
    def __len__(self):
        return len(self.X)

def get_loader(dataset_dir, dataset_type='train', batch_size=16, shuffle=True, limit=None):
    X_files = sorted(
        glob.glob(os.path.join(dataset_dir, dataset_type, 'images/**', '*.png'), recursive=True)
    )[:limit]
    y_files = sorted(
        glob.glob(os.path.join(dataset_dir, dataset_type, 'labels/**', '*.png'), recursive=True)
    )[:limit]

    dataset = Dataset2d(X_files, y_files)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    return loader