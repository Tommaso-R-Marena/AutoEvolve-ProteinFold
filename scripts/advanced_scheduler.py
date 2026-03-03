#!/usr/bin/env python3
"""Advanced training scheduler with multi-objective optimization.

Features:
- Warmup learning rate
- Cosine annealing with restarts
- Gradient norm-based loss balancing
- Adaptive batch sizing
- Learning rate finder
- Mixed precision training
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional
import numpy as np
import math

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-6, max_lr: float = 1e-3):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.min_lr + (self.max_lr - self.min_lr) * self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr
    
    def get_last_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

class GradientNormBalancer:
    """Balance multiple loss components based on gradient norms.
    
    Ensures all loss components contribute equally to training.
    Based on: Chen et al. (2018) "GradNorm: Gradient Normalization for Adaptive Loss Balancing"
    """
    
    def __init__(self, num_losses: int, alpha: float = 1.5):
        self.num_losses = num_losses
        self.alpha = alpha
        self.weights = torch.ones(num_losses)
        self.initial_losses = None
    
    def update_weights(self, model: nn.Module, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update loss weights based on gradient norms."""
        loss_list = list(losses.values())
        loss_names = list(losses.keys())
        
        # Track initial losses
        if self.initial_losses is None:
            self.initial_losses = torch.tensor([l.item() for l in loss_list])
        
        # Compute gradient norms
        grad_norms = []
        for loss in loss_list:
            # Compute gradient for this loss
            grads = torch.autograd.grad(loss, model.parameters(), 
                                       retain_graph=True, create_graph=False)
            grad_norm = torch.norm(torch.stack([torch.norm(g) for g in grads if g is not None]))
            grad_norms.append(grad_norm.item())
        
        grad_norms = torch.tensor(grad_norms)
        
        # Compute relative loss ratios
        current_losses = torch.tensor([l.item() for l in loss_list])
        loss_ratios = current_losses / (self.initial_losses + 1e-8)
        
        # Compute mean gradient norm
        mean_grad_norm = grad_norms.mean()
        
        # Target gradient norms (based on relative loss decrease)
        target_grad_norms = mean_grad_norm * (loss_ratios ** self.alpha)
        
        # Update weights to match target gradient norms
        self.weights = self.weights * (target_grad_norms / (grad_norms + 1e-8))
        self.weights = self.weights / self.weights.sum() * self.num_losses
        
        return {name: float(self.weights[i]) for i, name in enumerate(loss_names)}

class AdaptiveBatchSizer:
    """Dynamically adjust batch size based on gradient noise."""
    
    def __init__(self, initial_batch_size: int, min_batch_size: int = 1, 
                 max_batch_size: int = 32):
        self.batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.gradient_history = []
    
    def update(self, gradients: List[torch.Tensor]) -> int:
        """Update batch size based on gradient statistics."""
        # Compute gradient norm
        grad_norm = torch.norm(torch.stack([torch.norm(g) for g in gradients if g is not None]))
        self.gradient_history.append(grad_norm.item())
        
        # Keep recent history
        if len(self.gradient_history) > 100:
            self.gradient_history.pop(0)
        
        if len(self.gradient_history) < 10:
            return self.batch_size
        
        # Compute gradient variance (noise)
        grad_mean = np.mean(self.gradient_history[-10:])
        grad_std = np.std(self.gradient_history[-10:])
        noise_ratio = grad_std / (grad_mean + 1e-8)
        
        # High noise -> increase batch size
        # Low noise -> can decrease batch size
        if noise_ratio > 0.5:
            self.batch_size = min(self.batch_size * 2, self.max_batch_size)
        elif noise_ratio < 0.2:
            self.batch_size = max(self.batch_size // 2, self.min_batch_size)
        
        return self.batch_size

class LearningRateFinder:
    """Find optimal learning rate using the method from Leslie Smith."""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                 criterion: nn.Module, device: str = 'cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
    
    def find(self, train_loader, min_lr: float = 1e-7, max_lr: float = 1.0, 
            num_iter: int = 100) -> Dict:
        """Run LR range test.
        
        Returns:
            results: Dict with 'lrs' and 'losses' arrays
        """
        # Save initial model state
        initial_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        
        lrs = []
        losses = []
        
        # Exponentially increase LR
        lr_mult = (max_lr / min_lr) ** (1 / num_iter)
        lr = min_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # Training loop
        self.model.train()
        smooth_loss = None
        
        for i, batch in enumerate(train_loader):
            if i >= num_iter:
                break
            
            # Forward pass
            self.optimizer.zero_grad()
            sequences = batch['sequences'].to(self.device)
            targets = batch['coordinates'].to(self.device)
            
            predictions = self.model(sequences)
            loss = self.criterion(predictions, {'coordinates': targets}, batch['mask'])
            
            # Track loss
            lrs.append(lr)
            
            # Exponential moving average
            if smooth_loss is None:
                smooth_loss = loss.item()
            else:
                smooth_loss = 0.9 * smooth_loss + 0.1 * loss.item()
            losses.append(smooth_loss)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update learning rate
            lr *= lr_mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Stop if loss explodes
            if smooth_loss > losses[0] * 4:
                break
        
        # Restore initial model state
        self.model.load_state_dict(initial_state)
        
        # Find optimal LR (minimum of smoothed loss)
        min_idx = np.argmin(losses)
        optimal_lr = lrs[min_idx]
        
        return {
            'lrs': lrs,
            'losses': losses,
            'optimal_lr': optimal_lr,
            'suggested_max_lr': optimal_lr * 10
        }

class MixedPrecisionTrainer:
    """Mixed precision training wrapper."""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, enabled: bool = True):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler(enabled=enabled)
        self.enabled = enabled
    
    def train_step(self, sequences: torch.Tensor, targets: Dict, 
                   mask: torch.Tensor, criterion: nn.Module) -> Dict:
        """Single training step with mixed precision."""
        self.optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast(enabled=self.enabled):
            predictions = self.model(sequences)
            loss, loss_dict = criterion(predictions, targets, mask)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Unscale gradients and clip
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss_dict

if __name__ == '__main__':
    print("Advanced training scheduler loaded.")
    print("Features: Warmup, Cosine annealing, Gradient balancing, LR finder, Mixed precision")
