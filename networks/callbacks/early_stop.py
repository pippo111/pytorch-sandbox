import numpy as np
import torch

class EarlyStop():
    def __init__(self, model, checkpoint, mode='min', label='score', tries=20):
        self.model = model
        self.checkpoint = checkpoint
        self.trial = 0
        self.label = label
        self.tries = tries
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best_score = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best_score = -np.Inf

    def on_epoch_end(self, score):
        if self.monitor_op(score, self.best_score):
            print(f"{self.label} improved, {self.best_score} -> {score}")
            print(f"{self.label} improved by {self.best_score - score}")
            print(f'Saving model: output/models/{self.checkpoint}.pt')
            torch.save(self.model.state_dict(), f'output/models/{self.checkpoint}.pt')

            self.best_score = score
            self.trial = 0
            
        else:
            self.trial += 1

            if self.trial > self.tries:
                print(f'Early stopping')
                
                return True

            print(f"{self.label} did not improved ({self.best_score}), {self.trial} / {self.tries}")

        return False
