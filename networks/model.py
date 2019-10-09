import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from networks import network
from networks import loss
from utils.image import calc_dist_map
from utils.common import calc_weights

class MyModel():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Device: {self.device}')

    def train(
        self,
        arch,
        epochs,
        train_loader,
        valid_loader,
        n_filters,
        loss_name,
        n_channels=1,
        n_classes=1,
        learning_rate=1e-3
    ):
        model = network.get(arch)(n_channels, n_filters, n_classes).to(self.device)
        loss_fn = loss.get(loss_name)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        history = {
            'losses': [],
            'val_losses': []
        }
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1} / {epochs}")
            start = time.time()

            losses = []
            val_losses = []
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                model.train()
                optimizer.zero_grad()

                y_hat = model(X_batch)

                loss_val = loss_fn(y_hat, y_batch)

                loss_val.backward()
                optimizer.step()

                losses.append(loss_val.item())
                
            with torch.no_grad():
                for X_batch, y_batch in valid_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    model.eval()
                    
                    y_hat = model(X_batch)

                    loss_val = loss_fn(y_hat, y_batch)
                    
                    val_losses.append(loss_val.item())

            avg_loss = np.mean(losses)
            avg_val_loss = np.mean(val_losses)

            history['losses'].append(avg_loss)
            history['val_losses'].append(avg_val_loss)
            
            print(f'Time per epoch: {(time.time() - start):.3f} seconds')
            print(f'Train. loss:', avg_loss)
            print(f'Valid. loss:', avg_val_loss)
            print(f'---------------------------------------------------')

        return history