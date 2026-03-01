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
import traceback

sys.path.append(str(Path(__file__).parent.parent))

from model.architecture import EvolvableProteinFoldingModel
from model.data_generator import ProteinDataGenerator

class ProteinFoldingLoss(nn.Module):
    """Combined loss for protein folding prediction with robust error handling."""
    
    def __init__(self, coord_weight=1.0, angle_weight=0.5, confidence_weight=0.3):
        super().__init__()
        self.coord_weight = coord_weight
        self.angle_weight = angle_weight
        self.confidence_weight = confidence_weight
    
    def forward(self, predictions, targets, mask):
        # Coordinate loss (RMSD) with numerical stability
        try:
            coord_diff = (predictions['coordinates'] - targets['coordinates']) ** 2
            coord_diff = coord_diff * mask.unsqueeze(-1)
            coord_loss = torch.sqrt(torch.sum(coord_diff, dim=-1) + 1e-8).mean()
            
            # Clamp to prevent extreme values
            coord_loss = torch.clamp(coord_loss, 0, 1000)
        except Exception as e:
            print(f"⚠️  Warning: Coordinate loss error: {e}. Using fallback.")
            coord_loss = torch.tensor(0.0, device=predictions['coordinates'].device)
        
        # Angle loss with stability
        try:
            angle_loss = nn.functional.mse_loss(
                predictions['angles'] * mask.unsqueeze(-1),
                targets['angles'] * mask.unsqueeze(-1)
            )
            angle_loss = torch.clamp(angle_loss, 0, 1000)
        except Exception as e:
            print(f"⚠️  Warning: Angle loss error: {e}. Using fallback.")
            angle_loss = torch.tensor(0.0, device=predictions['angles'].device)
        
        # Confidence loss with value checking
        try:
            # Calculate coordinate error for target confidence
            coord_error = torch.sqrt(torch.sum(
                (predictions['coordinates'] - targets['coordinates']) ** 2,
                dim=-1
            ) + 1e-8)
            
            # Target confidence based on error (lower error = higher confidence)
            target_confidence = torch.exp(-coord_error / 5.0)
            
            # Clamp predictions to valid range [0, 1]
            pred_confidence = torch.clamp(predictions['confidence'], 0.0, 1.0)
            target_confidence = torch.clamp(target_confidence, 0.0, 1.0)
            
            # Check for NaN or Inf
            if torch.isnan(pred_confidence).any() or torch.isinf(pred_confidence).any():
                print("⚠️  Warning: NaN or Inf in predicted confidence. Skipping confidence loss.")
                confidence_loss = torch.tensor(0.0, device=pred_confidence.device)
            elif torch.isnan(target_confidence).any() or torch.isinf(target_confidence).any():
                print("⚠️  Warning: NaN or Inf in target confidence. Skipping confidence loss.")
                confidence_loss = torch.tensor(0.0, device=pred_confidence.device)
            else:
                confidence_loss = nn.functional.binary_cross_entropy(
                    pred_confidence,
                    target_confidence,
                    reduction='mean'
                )
                confidence_loss = torch.clamp(confidence_loss, 0, 10)
        except Exception as e:
            print(f"⚠️  Warning: Confidence loss error: {e}. Using fallback.")
            confidence_loss = torch.tensor(0.0, device=predictions['confidence'].device)
        
        # Combine losses with error checking
        try:
            total_loss = (
                self.coord_weight * coord_loss +
                self.angle_weight * angle_loss +
                self.confidence_weight * confidence_loss
            )
            
            # Final safety check
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("⚠️  Warning: Total loss is NaN or Inf. Using fallback loss.")
                total_loss = torch.tensor(10.0, device=total_loss.device, requires_grad=True)
        except Exception as e:
            print(f"⚠️  Warning: Loss combination error: {e}. Using fallback.")
            total_loss = torch.tensor(10.0, requires_grad=True)
        
        return total_loss, {
            'coord_loss': coord_loss.item() if torch.is_tensor(coord_loss) else 0.0,
            'angle_loss': angle_loss.item() if torch.is_tensor(angle_loss) else 0.0,
            'confidence_loss': confidence_loss.item() if torch.is_tensor(confidence_loss) else 0.0
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
        try:
            state = {
                'epoch': epoch,
                'optimizer_state': optimizer_state,
                'scheduler_state': scheduler_state,
                'total_samples': total_samples,
                'best_loss': best_loss,
                'rng_state': rng_state,
                'timestamp': time.time()
            }
            
            # Atomic write
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(state, f)
            temp_file.replace(self.state_file)
        except Exception as e:
            print(f"⚠️  Warning: Failed to save training state: {e}")
    
    def load_state(self):
        """Load training state if exists."""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Failed to load training state: {e}")
            # Try to load from backup
            backup_file = self.state_file.with_suffix('.bak')
            if backup_file.exists():
                try:
                    with open(backup_file, 'rb') as f:
                        return pickle.load(f)
                except:
                    pass
            return None
    
    def clear_state(self):
        """Clear training state."""
        try:
            if self.state_file.exists():
                # Create backup before clearing
                backup_file = self.state_file.with_suffix('.bak')
                if self.state_file.exists():
                    self.state_file.replace(backup_file)
                self.state_file.unlink(missing_ok=True)
        except Exception as e:
            print(f"⚠️  Warning: Failed to clear training state: {e}")

def train_cycle(args):
    """Run a training cycle with comprehensive error handling."""
    training_successful = False
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {device}")
        
        # Training state manager
        state_manager = TrainingState()
        
        # Load or create model
        checkpoint_path = Path('weights/latest.pt')
        config_path = Path('config/model_config.json')
        
        if checkpoint_path.exists():
            try:
                print("📦 Loading existing model...")
                model = EvolvableProteinFoldingModel.load_checkpoint(str(checkpoint_path))
                print(f"✅ Loaded model at generation {model.generation}")
            except Exception as e:
                print(f"⚠️  Warning: Failed to load checkpoint: {e}")
                print("🔨 Creating new model instead...")
                with open(config_path) as f:
                    config = json.load(f)
                model = EvolvableProteinFoldingModel(config)
        else:
            print("🔨 Creating new model...")
            with open(config_path) as f:
                config = json.load(f)
            model = EvolvableProteinFoldingModel(config)
        
        model = model.to(device)
        
        # Get max sequence length from config
        max_seq_len = model.config.get('max_sequence_length', 256)
        
        # Data generator
        data_generator = ProteinDataGenerator()
        
        # Try to fetch real data from UniProt
        print("\n🌐 Attempting to fetch real protein data...")
        try:
            real_data = data_generator.fetch_real_data_uniprot(n_samples=50)
            if real_data:
                print(f"✅ Fetched {len(real_data)} real protein sequences from UniProt")
                data_info_path = Path('data/training_data_info.json')
                data_info_path.parent.mkdir(exist_ok=True, parents=True)
                with open(data_info_path, 'w') as f:
                    json.dump({
                        'real_sequences': len(real_data),
                        'last_updated': time.time()
                    }, f)
            else:
                print("⚠️  No real data fetched, using synthetic data only")
        except Exception as e:
            print(f"⚠️  Warning: Failed to fetch real data: {e}")
            print("🔬 Using synthetic data only")
        
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
                try:
                    print(f"\n♻️  Resuming from epoch {saved_state['epoch']}")
                    start_epoch = saved_state['epoch']
                    total_samples_seen = saved_state['total_samples']
                    best_loss = saved_state['best_loss']
                    
                    optimizer.load_state_dict(saved_state['optimizer_state'])
                    scheduler.load_state_dict(saved_state['scheduler_state'])
                    
                    # Restore RNG state for reproducibility
                    torch.set_rng_state(saved_state['rng_state']['torch'])
                    np.random.set_state(saved_state['rng_state']['numpy'])
                    
                    print(f"  📊 Total samples processed: {total_samples_seen:,}")
                    print(f"  🎯 Best loss so far: {best_loss:.4f}")
                except Exception as e:
                    print(f"⚠️  Warning: Failed to restore training state: {e}")
                    print("🔄 Starting fresh...")
        
        # Training loop
        start_time = time.time()
        max_time = args.max_time
        
        epoch = start_epoch
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        print(f"\n🚀 Starting training for {max_time} seconds...")
        print(f"💾 Will save state every 100 epochs for seamless resumption")
        print(f"📏 Max sequence length: {max_seq_len}\n")
        
        while time.time() - start_time < max_time and consecutive_errors < max_consecutive_errors:
            try:
                epoch += 1
                model.train()
                
                # Generate training batch with limited sequence length
                try:
                    batch = data_generator.generate_synthetic_batch(
                        args.batch_size,
                        min_len=30,
                        max_len=min(max_seq_len, 200)
                    )
                    total_samples_seen += args.batch_size
                except Exception as e:
                    print(f"⚠️  Warning: Batch generation failed: {e}. Retrying...")
                    consecutive_errors += 1
                    continue
                
                # Move to device
                sequences = batch['sequences'].to(device)
                target_coords = batch['coordinates'].to(device)
                mask = batch['mask'].to(device)
                
                # Synthetic angles
                target_angles = torch.randn_like(target_coords)
                
                # Forward pass with error handling
                optimizer.zero_grad()
                
                try:
                    predictions = model(sequences)
                    
                    # Sanity check predictions
                    if torch.isnan(predictions['coordinates']).any() or torch.isinf(predictions['coordinates']).any():
                        print(f"⚠️  Warning: NaN or Inf in predictions at epoch {epoch}. Skipping batch.")
                        consecutive_errors += 1
                        continue
                    
                except RuntimeError as e:
                    if "allocate memory" in str(e) or "out of memory" in str(e).lower():
                        print(f"\n⚠️  Memory error at epoch {epoch}. Reducing batch size...")
                        args.batch_size = max(1, args.batch_size // 2)
                        consecutive_errors += 1
                        continue
                    else:
                        print(f"⚠️  Runtime error during forward pass: {e}")
                        consecutive_errors += 1
                        continue
                except Exception as e:
                    print(f"⚠️  Unexpected error during forward pass: {e}")
                    consecutive_errors += 1
                    continue
                
                # Compute loss
                targets = {
                    'coordinates': target_coords,
                    'angles': target_angles
                }
                
                try:
                    loss, loss_dict = criterion(predictions, targets, mask)
                    
                    # Check loss validity
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 10000:
                        print(f"⚠️  Warning: Invalid loss at epoch {epoch}: {loss.item()}. Skipping batch.")
                        consecutive_errors += 1
                        continue
                    
                except Exception as e:
                    print(f"⚠️  Error computing loss: {e}")
                    print(f"    Traceback: {traceback.format_exc()}")
                    consecutive_errors += 1
                    continue
                
                # Backward pass with error handling
                try:
                    loss.backward()
                    
                    # Check for gradient explosion
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if total_norm > 100:
                        print(f"⚠️  Warning: Large gradient norm ({total_norm:.2f}) at epoch {epoch}")
                    
                    optimizer.step()
                    scheduler.step()
                    
                    # Reset error counter on success
                    consecutive_errors = 0
                    
                except Exception as e:
                    print(f"⚠️  Error during backward pass: {e}")
                    consecutive_errors += 1
                    continue
                
                # Logging
                if epoch % 10 == 0:
                    elapsed = time.time() - start_time
                    remaining = max_time - elapsed
                    print(f"📈 Epoch {epoch} | Loss: {loss.item():.4f} | "
                          f"Coord: {loss_dict['coord_loss']:.4f} | "
                          f"Samples: {total_samples_seen:,} | "
                          f"⏱️  {elapsed:.0f}s | Remaining: {remaining:.0f}s")
                
                # Save checkpoint periodically
                if loss.item() < best_loss and epoch % 50 == 0:
                    try:
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
                        print(f"  💾 Saved checkpoint with loss {loss.item():.4f}")
                    except Exception as e:
                        print(f"⚠️  Warning: Failed to save checkpoint: {e}")
                
                # Save training state every 100 epochs
                if epoch % 100 == 0:
                    try:
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
                    except Exception as e:
                        print(f"⚠️  Warning: Failed to save training state: {e}")
            
            except KeyboardInterrupt:
                print("\n\n⏸️  Training interrupted by user. Saving state...")
                break
            except Exception as e:
                print(f"\n⚠️  Unexpected error at epoch {epoch}: {e}")
                print(traceback.format_exc())
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"\n❌ Too many consecutive errors ({consecutive_errors}). Stopping training.")
                    break
        
        # Determine if training was successful
        training_successful = consecutive_errors < max_consecutive_errors
        
        # Check if we stopped due to errors
        if not training_successful:
            print(f"\n⚠️  Training stopped due to {consecutive_errors} consecutive errors.")
        else:
            print(f"\n✅ Training completed successfully!")
        
        # Final save
        print("\n💾 Saving final state...")
        try:
            model.save_checkpoint(str(checkpoint_path), {
                'final_epoch': epoch,
                'final_loss': best_loss,
                'total_time': time.time() - start_time,
                'total_samples': total_samples_seen,
                'completed': training_successful
            })
        except Exception as e:
            print(f"⚠️  Warning: Failed to save final checkpoint: {e}")
        
        # Save final training state
        try:
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
        except Exception as e:
            print(f"⚠️  Warning: Failed to save final training state: {e}")
        
        # Save metrics
        try:
            metrics_path = Path('metrics/training_metrics.json')
            metrics_path.parent.mkdir(exist_ok=True)
            
            metrics = {
                'generation': model.generation,
                'epochs': epoch,
                'final_loss': best_loss,
                'best_loss': best_loss,
                'training_time': time.time() - start_time,
                'total_samples': total_samples_seen,
                'samples_per_epoch': total_samples_seen / epoch if epoch > 0 else 0,
                'completed_successfully': training_successful
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"📊 Metrics saved to {metrics_path}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to save metrics: {e}")
        
        print(f"\n{'='*60}")
        print("📊 Training Summary:")
        print(f"{'='*60}")
        print(f"  🔢 Epochs: {epoch} (started from {start_epoch})")
        print(f"  📦 Samples processed: {total_samples_seen:,}")
        print(f"  ⏱️  Training time: {time.time() - start_time:.1f}s")
        print(f"  🎯 Best loss: {best_loss:.4f}")
        print(f"  ✅ Status: {'Completed' if training_successful else 'Stopped (errors)'}")
        print(f"{'='*60}\n")
        
        # Exit with appropriate code
        if not training_successful:
            sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Fatal error in training cycle: {e}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='continuous')
    parser.add_argument('--max-time', type=int, default=18000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--resume', action='store_true',
                       help='Resume from saved training state')
    args = parser.parse_args()
    
    try:
        train_cycle(args)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print(traceback.format_exc())
        sys.exit(1)
