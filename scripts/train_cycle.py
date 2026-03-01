#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pickle

sys.path.append(str(Path(__file__).parent.parent))

from model.architecture import EvolvableProteinFoldingModel
from model.data_generator import ProteinDataGenerator

class ProteinFoldingLoss(nn.Module):
    """Combined loss for protein folding prediction."""
    
    def __init__(self, coord_weight=1.0, angle_weight=0.5, confidence_weight=0.3):
        super().__init__()
        self.coord_weight = coord_weight
        self.angle_weight = angle_weight
        self.confidence_weight = confidence_weight
    
    def forward(self, predictions, targets, mask):
        # Coordinate loss (RMSD)
        coord_loss = torch.sqrt(torch.sum(
            (predictions['coordinates'] - targets['coordinates']) ** 2 * mask.unsqueeze(-1),
            dim=-1
        )).mean()
        
        # Angle loss
        angle_loss = nn.functional.mse_loss(
            predictions['angles'] * mask.unsqueeze(-1),
            targets['angles'] * mask.unsqueeze(-1)
        )
        
        # Confidence loss
        coord_error = torch.sqrt(torch.sum(
            (predictions['coordinates'] - targets['coordinates']) ** 2,
            dim=-1
        ))
        target_confidence = torch.exp(-coord_error / 5.0)
        confidence_loss = nn.functional.binary_cross_entropy(
            predictions['confidence'],
            target_confidence
        )
        
        total_loss = (
            self.coord_weight * coord_loss +
            self.angle_weight * angle_loss +
            self.confidence_weight * confidence_loss
        )
        
        return total_loss, {
            'coord_loss': coord_loss.item(),
            'angle_loss': angle_loss.item(),
            'confidence_loss': confidence_loss.item()
        }

class TrainingState:
    """Manage training state for seamless resumption."""
    
    def __init__(self, state_dir: str = 'data/training_state'):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / 'training_state.pkl'
    
    def save_state(self, epoch: int, optimizer_state: dict, scheduler_state: dict,
                   total_samples: int, best_loss: float, rng_state: dict):
        """Save complete training state."""
        state = {
            'epoch': epoch,
            'optimizer_state': optimizer_state,
            'scheduler_state': scheduler_state,
            'total_samples': total_samples,
            'best_loss': best_loss,
            'rng_state': rng_state,
            'timestamp': time.time()
        }
        
        with open(self.state_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self):
        """Load training state if exists."""
        if not self.state_file.exists():
            return None
        
        with open(self.state_file, 'rb') as f:
            return pickle.load(f)
    
    def clear_state(self):
        """Clear training state."""
        if self.state_file.exists():
            self.state_file.unlink()

def train_cycle(args):
    """Run a training cycle with continuous improvement and state resumption."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Training state manager
    state_manager = TrainingState()
    
    # Load or create model
    checkpoint_path = Path('weights/latest.pt')
    config_path = Path('config/model_config.json')
    
    if checkpoint_path.exists():
        print("Loading existing model...")
        model = EvolvableProteinFoldingModel.load_checkpoint(str(checkpoint_path))
        print(f"Loaded model at generation {model.generation}")
    else:
        print("Creating new model...")
        with open(config_path) as f:
            config = json.load(f)
        model = EvolvableProteinFoldingModel(config)
    
    model = model.to(device)
    
    # Get max sequence length from config
    max_seq_len = model.config.get('max_sequence_length', 256)
    
    # Data generator
    data_generator = ProteinDataGenerator()
    
    # Try to fetch real data from UniProt
    print("\nAttempting to fetch real protein data...")
    real_data = data_generator.fetch_real_data_uniprot(n_samples=50)
    
    if real_data:
        print(f"✓ Fetched {len(real_data)} real protein sequences from UniProt")
        data_info_path = Path('data/training_data_info.json')
        data_info_path.parent.mkdir(exist_ok=True, parents=True)
        with open(data_info_path, 'w') as f:
            json.dump({
                'real_sequences': len(real_data),
                'last_updated': time.time()
            }, f)
    else:
        print("Using synthetic data only")
    
    # Optimizer with adaptive learning rate
    base_lr = 1e-4 * (0.95 ** model.generation)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    criterion = ProteinFoldingLoss()
    
    # Resume from saved state if requested
    start_epoch = 0
    total_samples_seen = 0
    best_loss = float('inf')
    
    if args.resume:
        saved_state = state_manager.load_state()
        if saved_state:
            print(f"\n✓ Resuming from epoch {saved_state['epoch']}")
            start_epoch = saved_state['epoch']
            total_samples_seen = saved_state['total_samples']
            best_loss = saved_state['best_loss']
            
            optimizer.load_state_dict(saved_state['optimizer_state'])
            scheduler.load_state_dict(saved_state['scheduler_state'])
            
            # Restore RNG state for reproducibility
            torch.set_rng_state(saved_state['rng_state']['torch'])
            np.random.set_state(saved_state['rng_state']['numpy'])
            
            print(f"  Total samples processed: {total_samples_seen:,}")
            print(f"  Best loss so far: {best_loss:.4f}")
    
    # Training loop
    start_time = time.time()
    max_time = args.max_time
    
    epoch = start_epoch
    
    print(f"\nStarting training for {max_time} seconds...")
    print(f"Will save state every 100 epochs for seamless resumption")
    print(f"Max sequence length: {max_seq_len}\n")
    
    while time.time() - start_time < max_time:
        epoch += 1
        model.train()
        
        # Generate training batch with limited sequence length
        batch = data_generator.generate_synthetic_batch(
            args.batch_size,
            min_len=30,
            max_len=min(max_seq_len, 200)  # Cap at max_seq_len or 200
        )
        total_samples_seen += args.batch_size
        
        # Move to device
        sequences = batch['sequences'].to(device)
        target_coords = batch['coordinates'].to(device)
        mask = batch['mask'].to(device)
        
        # Synthetic angles
        target_angles = torch.randn_like(target_coords)
        
        # Forward pass
        optimizer.zero_grad()
        
        try:
            predictions = model(sequences)
        except RuntimeError as e:
            if "allocate memory" in str(e):
                print(f"\n⚠️  Memory error at epoch {epoch}. Reducing batch size and continuing...")
                # Reduce batch size for this iteration
                args.batch_size = max(1, args.batch_size // 2)
                continue
            else:
                raise
        
        # Compute loss
        targets = {
            'coordinates': target_coords,
            'angles': target_angles
        }
        
        loss, loss_dict = criterion(predictions, targets, mask)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Logging
        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            remaining = max_time - elapsed
            print(f"Epoch {epoch} | Loss: {loss.item():.4f} | "
                  f"Coord: {loss_dict['coord_loss']:.4f} | "
                  f"Samples: {total_samples_seen:,} | "
                  f"Time: {elapsed:.0f}s | Remaining: {remaining:.0f}s")
        
        # Save checkpoint periodically
        if loss.item() < best_loss and epoch % 50 == 0:
            best_loss = loss.item()
            metadata = {
                'epoch': epoch,
                'loss': loss.item(),
                'timestamp': time.time(),
                'total_samples': total_samples_seen
            }
            model.performance_history.append(metadata)
            
            checkpoint_path.parent.mkdir(exist_ok=True)
            model.save_checkpoint(str(checkpoint_path), metadata)
            print(f"  → Saved checkpoint with loss {loss.item():.4f}")
        
        # Save training state every 100 epochs
        if epoch % 100 == 0:
            state_manager.save_state(
                epoch=epoch,
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                total_samples=total_samples_seen,
                best_loss=best_loss,
                rng_state={
                    'torch': torch.get_rng_state(),
                    'numpy': np.random.get_state()
                }
            )
    
    # Final save
    print("\nTraining cycle complete. Saving final state...")
    model.save_checkpoint(str(checkpoint_path), {
        'final_epoch': epoch,
        'final_loss': loss.item(),
        'total_time': time.time() - start_time,
        'total_samples': total_samples_seen
    })
    
    # Save final training state
    state_manager.save_state(
        epoch=epoch,
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        total_samples=total_samples_seen,
        best_loss=best_loss,
        rng_state={
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state()
        }
    )
    
    # Save metrics
    metrics_path = Path('metrics/training_metrics.json')
    metrics_path.parent.mkdir(exist_ok=True)
    
    metrics = {
        'generation': model.generation,
        'epochs': epoch,
        'final_loss': loss.item(),
        'best_loss': best_loss,
        'training_time': time.time() - start_time,
        'total_samples': total_samples_seen,
        'samples_per_epoch': total_samples_seen / epoch if epoch > 0 else 0
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training Summary:")
    print(f"{'='*60}")
    print(f"  Epochs: {epoch} (started from {start_epoch})")
    print(f"  Samples processed: {total_samples_seen:,}")
    print(f"  Training time: {time.time() - start_time:.1f}s")
    print(f"  Final loss: {loss.item():.4f}")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='continuous')
    parser.add_argument('--max-time', type=int, default=18000)
    parser.add_argument('--batch-size', type=int, default=4)  # Reduced from 8
    parser.add_argument('--resume', action='store_true',
                       help='Resume from saved training state')
    args = parser.parse_args()
    
    train_cycle(args)
