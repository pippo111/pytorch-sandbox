import numpy as np
import torch

class EarlyStop():
    def __init__(self, model, checkpoint, tries=20):
        self.model = model
        self.checkpoint = checkpoint
        self.best_score = np.Inf
        self.trial = 0
        self.tries = tries

    def on_epoch_end(self, score):
        if score < self.best_score:
            print(f"val_loss improved, {self.best_score} -> {score}")
            print(f"val_loss improved by {self.best_score - score}")
            print(f'Saving model: output/models/{self.checkpoint}.pt')
            torch.save(self.model.state_dict(), f'output/models/{self.checkpoint}.pt')

            self.best_score = score
            self.trial = 0
            
        else:
            self.trial += 1

            if self.trial > self.tries:
                print(f'Early stopping')
                
                return True

            print(f"val_loss did not improved ({score}), {self.trial} / {self.tries}")

        return False
