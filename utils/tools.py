import numpy as np
import torch
import os

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, min_delta=1e-4):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.min_delta = min_delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(val_loss, model, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(val_loss, model, path)
            self.counter = 0

    def _save(self, val_loss, model, path):
        if self.verbose:
            print(f'Val loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving...')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
