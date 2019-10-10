import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from networks import network
from networks import loss
from utils.image import calc_dist_map
from utils.common import calc_weights
from utils.metrics import calc_confusion_matrix, calc_fn_rate, calc_fp_rate

class MyModel():
    def __init__(self, struct):
        self.struct = struct
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f'Device: {self.device}')
        print(f'---------------------------------------------------')

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
        learning_rate=1e-3,
        lr_patience=10,
        tries=20
    ):
        model = network.get(arch)(n_channels, n_filters, n_classes).to(self.device)
        loss_fn = loss.get(loss_name)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=lr_patience, verbose=True)

        checkpoint = "{}_{}_{}_{}_bs-{}_f-{}".format(
                    self.struct,
                    arch,
                    'Adam',
                    loss_name,
                    16,
                    n_filters
                )

        history = {
            'losses': [],
            'val_losses': [],
            'dices': [],
            'val_dices': [],
            'fp_rate': [],
            'fn_rate': [],
            'f_total': []
        }

        best_score = np.Inf
        trial = 0
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1} / {epochs}")
            start = time.time()
            best_f_total = np.Inf

            losses = []
            val_losses = []
            dices = []
            val_dices = []
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                model.train()
                optimizer.zero_grad()

                y_hat = model(X_batch)

                loss_val = loss_fn(y_hat, y_batch)
                dice_val = loss.get('dice')(y_hat, y_batch)

                loss_val.backward()
                optimizer.step()

                losses.append(loss_val.item())
                dices.append(dice_val.item())
                
            with torch.no_grad():
                confusions = dict(
                    fp_total = 0,
                    fn_total = 0,
                    tp_total = 0,
                    tn_total = 0,
                    f_total = 0
                )

                for X_batch, y_batch in valid_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    model.eval()
                    
                    y_hat = model(X_batch)

                    loss_val = loss_fn(y_hat, y_batch)
                    dice_val = loss.get('dice')(y_hat, y_batch)

                    # Calc confusion matrix on the fly
                    np_y_batch = y_batch.cpu().numpy()
                    np_y_hat = y_hat.cpu().numpy()
                    np_y_pred = (np_y_hat > 0.5).astype(np.uint8)
                    conf = calc_confusion_matrix(np_y_batch, np_y_pred)
                    
                    confusions['fp_total'] += conf['fp_total']
                    confusions['fn_total'] += conf['fn_total']
                    confusions['tp_total'] += conf['tp_total']
                    confusions['tn_total'] += conf['tn_total']
                    confusions['f_total'] += conf['f_total']
                    
                    val_losses.append(loss_val.item())
                    val_dices.append(dice_val.item())

                fpr_perc = calc_fp_rate(confusions['fp_total'], confusions['tn_total'])
                fnr_perc = calc_fn_rate(confusions['fn_total'], confusions['tp_total'])

            avg_loss = np.mean(losses)
            avg_val_loss = np.mean(val_losses)
            avg_dice = np.mean(dices)
            avg_val_dice = np.mean(val_dices)

            history['losses'].append(avg_loss)
            history['val_losses'].append(avg_val_loss)
            history['dices'].append(avg_dice)
            history['val_dices'].append(avg_val_dice)
            history['fp_rate'].append(fpr_perc)
            history['fn_rate'].append(fnr_perc)
            history['f_total'].append(confusions['f_total'])
            
            print(f'Time per epoch: {(time.time() - start):.3f} seconds')
            print(f'---')
            print(f'Train. loss:', avg_loss)
            print(f'Valid. loss:', avg_val_loss)
            print(f'---')
            print(f'Train. dice:', avg_dice)
            print(f'Valid. dice:', avg_val_dice)
            print(f'---')
            print(f'False positive rate: {fpr_perc}')
            print(f'False negative rate: {fnr_perc}')
            print(f'---')
            print(f"FP+FN:, {confusions['f_total']}")
            print(f'---')

            if avg_val_loss < best_score:
                print(f"val_loss improved, {best_score} -> {avg_val_loss}")
                print(f"val_loss improved by {best_score - avg_val_loss}")
                print(f'Saving model: output/models/{checkpoint}.pt')
                torch.save(model.state_dict(), f'output/models/{checkpoint}.pt')

                best_score = avg_val_loss
                trial = 0
                
            else:
                trial += 1

                if trial > tries:
                    print(f'Early stopping')
                    break

                print(f"val_loss did not improved ({avg_val_loss}), {trial} / {tries}")

            scheduler.step(avg_val_loss)
            print(f'---------------------------------------------------')

        return history