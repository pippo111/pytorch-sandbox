import time
import os
import operator
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from networks import network
from networks import loss
from networks import optimizer
from networks.callbacks.early_stop import EarlyStop
from utils.image import calc_dist_map, cubify_scan, labels_to_mask
from utils.common import calc_weights
from utils.metrics import calc_confusion_matrix, calc_fn_rate, calc_fp_rate, calc_precision, calc_recall, calc_f1score
from utils.vtk import render_mesh
from utils.logs import to_table

class MyModel():
    def __init__(
        self,
        arch,
        struct,
        n_filters,
        batch_size=16,
        n_channels=1,
        n_classes=1
    ):
        self.struct = struct
        self.arch = arch
        self.n_filters = n_filters
        self.batch_size = batch_size

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = network.get(arch)(n_channels, n_filters, n_classes).to(self.device)

        self.history = {
            'losses': [],
            'val_losses': [],
            'dices': [],
            'val_dices': [],
            'fp_rate': [],
            'fn_rate': [],
            'fp_total': [],
            'fn_total': [],
            'f_total': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'time_per_epoch': []
        }

        print(f'Device: {self.device}')
        print(f'---------------------------------------------------')

    def load(self, filename):
        self.model.load_state_dict(torch.load(f'output/models/{filename}'))
        self.model.eval()

    def train(
        self,
        epochs, 
        train_loader,
        valid_loader,
        loss_name,
        optimizer_name='Adam',
        learning_rate=1e-3,
        lr_patience=10,
        tries=20
    ):
        self.checkpoint = "{}_{}_{}_{}_bs-{}_f-{}".format(
            self.struct,
            self.arch,
            optimizer_name,
            loss_name,
            self.batch_size,
            self.n_filters
        )

        self.loss_name = loss_name
        self.optimizer_name = optimizer_name

        loss_fn = loss.get(loss_name)
        optimizer_fn = optimizer.get(optimizer_name)(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_fn, mode='min', factor=0.5, patience=lr_patience, verbose=True)
        early_stop = EarlyStop(self.model, self.checkpoint, mode='min', label='Falses total', tries=tries)
        alpha_step = 1 / epochs
        alpha_init = 1.0
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1} / {epochs}")
            start = time.time()

            losses = []
            val_losses = []
            dices = []
            val_dices = []
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.model.train()
                optimizer_fn.zero_grad()

                y_hat = self.model(X_batch)

                if loss_name.startswith('boundary'):
                    alpha = alpha_init - epoch * alpha_step
                    loss_val = loss_fn(y_hat, y_batch, alpha)
                else:
                    loss_val = loss_fn(y_hat, y_batch)

                dice_val = loss.get('dice')(y_hat, y_batch)

                loss_val.backward()
                optimizer_fn.step()

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
                    
                    self.model.eval()
                    
                    y_hat = self.model(X_batch)

                    if loss_name.startswith('boundary'):
                        alpha = alpha_init - epoch * alpha_step
                        loss_val = loss_fn(y_hat, y_batch, alpha)
                    else:
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

            time_per_epoch = time.time() - start

            self.log_history(time_per_epoch, losses, val_losses, dices, val_dices, confusions)
            self.last_step_stats()
            
            if early_stop.on_epoch_end(score = confusions['f_total']):
                break

            scheduler.step(confusions['f_total'])

            print(f'---------------------------------------------------')

        np.save(f'output/models/{self.checkpoint}_history.npy', self.history)

        return self.history

    def visualize(self, test_loader):
        scan_preds = list()
        scan_mask = list()

        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            with torch.no_grad():
                self.model.eval()

                preds = self.model(X_batch)
                preds = preds.cpu().numpy()
                preds = (preds > 0.5).astype(np.uint8)
                scan_preds.append(preds.squeeze())
                
                mask = y_batch.cpu().numpy().squeeze().astype(np.uint8)
                scan_mask.append(mask)
                
        scan_preds = np.concatenate([img for img in scan_preds])
        scan_mask = np.concatenate([img for img in scan_mask])

        print('Preds shape:', scan_preds.shape)
        print('Mask shape:', scan_mask.shape)

        combined_scan = scan_mask * 2 + scan_preds

        false_positive = cubify_scan(labels_to_mask(combined_scan, [1]), 256)
        false_negative = cubify_scan(labels_to_mask(combined_scan, [2]), 256)
        true_positive = cubify_scan(labels_to_mask(combined_scan, [3]), 256)

        render_mesh([
            {
                'data': false_negative,
                'color': 'Crimson',
                'opacity': 0.6
            },
            {
                'data': false_positive,
                'color': 'Gold',
                'opacity': 0.6
            },
            {
                'data': true_positive,
                'color': 'ForestGreen',
                'opacity': 1.0
            }
        ], 256)

    def save_results(self):
        csv_file = f'output/models/{self.struct}_results.csv'
        html_file = f'output/models/{self.struct}_results.html'

        index, value = min(enumerate(self.history['f_total']), key=operator.itemgetter(1))

        setup = {
            'Arch': self.arch,
            'Optimizer': self.optimizer_name,
            'Loss fn': self.loss_name,
            'Batch size': self.batch_size,
            'Filters': self.n_filters
        }

        results = {
            'Time per epoch': f"{self.history['time_per_epoch'][index]:.3f}",
            'Train. loss': self.history['losses'][index],
            'Valid. loss': self.history['val_losses'][index],
            'Train. dice': self.history['dices'][index],
            'Valid. dice': self.history['val_dices'][index],
            'False positive rate': f"{self.history['fp_rate'][index]:.2%}",
            'False negative rate': f"{self.history['fn_rate'][index]:.2%}",
            'Precision rate': f"{self.history['precision'][index]:.2%}",
            'Recall rate': f"{self.history['recall'][index]:.2%}",
            'F1 score rate': f"{self.history['f1_score'][index]:.2%}",
            'FP': self.history['fp_total'][index],
            'FN': self.history['fn_total'][index],
            'FP+FN': self.history['f_total'][index],
        }

        results = [{ 'checkpoint': self.checkpoint, **setup, **results }]
        output = pd.DataFrame(results)

        if not os.path.exists(csv_file):
            output.to_csv(csv_file, index=False, header=True, mode='a')
        else:
            output.to_csv(csv_file, index=False, header=False, mode='a')

        generated_csv = pd.read_csv(csv_file)

        to_table(generated_csv.to_html(index=False), html_file)


    def log_history(self, time_per_epoch, losses, val_losses, dices, val_dices, confusions):
        avg_loss = np.mean(losses)
        avg_val_loss = np.mean(val_losses)
        avg_dice = np.mean(dices)
        avg_val_dice = np.mean(val_dices)

        fp_rate = calc_fp_rate(confusions['fp_total'], confusions['tn_total'])
        fn_rate = calc_fn_rate(confusions['fn_total'], confusions['tp_total'])
        precision = calc_precision(confusions['tp_total'], confusions['fp_total'])
        recall = calc_recall(confusions['tp_total'], confusions['fn_total'])
        f1_score = calc_f1score(confusions['tp_total'], confusions['fp_total'], confusions['fn_total'])

        self.history['time_per_epoch'].append(time_per_epoch)
        self.history['losses'].append(avg_loss)
        self.history['val_losses'].append(avg_val_loss)
        self.history['dices'].append(avg_dice)
        self.history['val_dices'].append(avg_val_dice)
        self.history['fp_rate'].append(fp_rate)
        self.history['fn_rate'].append(fn_rate)
        self.history['fp_total'].append(confusions['fp_total'])
        self.history['fn_total'].append(confusions['fn_total'])
        self.history['f_total'].append(confusions['f_total'])
        self.history['precision'].append(precision)
        self.history['recall'].append(recall)
        self.history['f1_score'].append(f1_score)

    def last_step_stats(self):
        print(f"Time per epoch: {self.history['time_per_epoch'][-1]:.3f} seconds")
        print(f'---')
        print(f'Train. loss:', self.history['losses'][-1])
        print(f'Valid. loss:', self.history['val_losses'][-1])
        print(f'---')
        print(f'Train. dice:', self.history['dices'][-1])
        print(f'Valid. dice:', self.history['val_dices'][-1])
        print(f'---')
        print(f"False positive rate: {self.history['fp_rate'][-1]:.2%}")
        print(f"False negative rate: {self.history['fn_rate'][-1]:.2%}")
        print(f"Precision rate: {self.history['precision'][-1]:.2%}")
        print(f"Recall rate: {self.history['recall'][-1]:.2%}")
        print(f"F1 score rate: {self.history['f1_score'][-1]:.2%}")
        print(f'---')
        print(f"FP: {self.history['fp_total'][-1]}")
        print(f"FN: {self.history['fn_total'][-1]}")
        print(f"FP+FN:, {self.history['f_total'][-1]}")
        print(f'---')

