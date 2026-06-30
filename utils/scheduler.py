import numpy as np
import torch.optim as optim

class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler that implements linear warmup followed by cosine annealing.
    """
    def __init__(self, optimizer: optim.Optimizer, warmup_epochs: int, 
                 total_epochs: int, min_lr: float = 1e-6, last_epoch: int = -1):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Number of linear warmup epochs.
            total_epochs (int): Total training epochs.
            min_lr (float): Minimum learning rate decay value.
            last_epoch (int): The index of last epoch.
        """
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup:
            # Linear warmup
            return [base_lr * (e + 1) / self.warmup for base_lr in self.base_lrs]
        
        # Cosine annealing
        progress = (e - self.warmup) / max(1, self.total - self.warmup)
        factor = max(0.5 * (1.0 + np.cos(np.pi * progress)),
                     self.min_lr / self.base_lrs[0])
        return [base_lr * factor for base_lr in self.base_lrs]
