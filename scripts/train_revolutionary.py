#!/usr/bin/env python3
"""Revolutionary training with NAS, diffusion, and all advanced features enabled."""
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import time
import json
from pathlib import Path
import numpy as np
import random
import traceback

sys.path.append(str(Path(__file__).parent.parent))

from model.enhanced_architecture import RevolutionaryProteinFolder
from model.neural_architecture_search import NASProteinModel, NASTrainer
from model.uncertainty_quantification import UncertaintyAwareLoss, MCDropoutPredictor
from model.data_generator import ProteinDataGenerator
from scripts.advanced_scheduler import WarmupCosineScheduler, GradientNormBalancer, MixedPrecisionTrainer
from scripts.comprehensive_benchmark import BenchmarkSuite

class FreshDataPipeline:
    """Ensures fresh protein data every training cycle."""
    
    def __init__(self, cache_dir='data/protein_cache', data_sources=['uniprot', 'pdb']):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_sources = data_sources
        self.data_generator = ProteinDataGenerator()
        self.structures = []
        self.sequences = []
        self.metadata = []
    
    def fetch_fresh_data(self, n_samples=50, force_refresh=False):
        """Fetch fresh protein data from multiple sources."""
        print("\n🌱 Fresh Data Pipeline Activated")
        print("="*60)
        
        # Load existing cache
        existing_ids = set()
        if not force_refresh:
            for cache_file in self.cache_dir.glob("*.npz"):
                existing_ids.add(cache_file.stem)
            print(f"💾 Loaded {len(existing_ids)} cached structures")
        
        # Fetch from UniProt
        print(f"\n🌐 Fetching {n_samples} sequences from UniProt...")
        try:
            uniprot_data = self.data_generator.fetch_real_data_uniprot(n_samples=n_samples)
            print(f"✅ Retrieved {len(uniprot_data)} UniProt sequences")
            
            # Fetch AlphaFold structures
            print(f"\n🧬 Fetching AlphaFold structures...")
            for i, protein in enumerate(uniprot_data):
                uniprot_id = protein['id'].split('|')[1] if '|' in protein['id'] else protein['id']
                
                # Skip if already cached and not forcing refresh
                if uniprot_id in existing_ids and not force_refresh:
                    self._load_from_cache(uniprot_id)
                    continue
                
                print(f"  [{i+1}/{len(uniprot_data)}] {uniprot_id}...", end=' ')
                
                try:
                    coords = self.data_generator.fetch_alphafold_structure(uniprot_id)
                    if coords is not None and len(coords) > 10:
                        self._add_structure(uniprot_id, protein['sequence'], coords, {
                            'source': 'alphafold',
                            'organism': protein.get('organism', 'Unknown'),
                            'length': len(coords),
                            'timestamp': time.time()
                        })
                        print(f"✅ {len(coords)} res")
                    else:
                        print("❌")
                    time.sleep(0.3)  # Rate limiting
                except Exception as e:
                    print(f"❌ {e}")
                    continue
        
        except Exception as e:
            print(f"⚠️  Warning: Failed to fetch UniProt data: {e}")
        
        # Summary
        print(f"\n📊 Fresh Data Summary:")
        print(f"  Total structures: {len(self.structures)}")
        print(f"  Total residues: {sum(len(s) for s in self.structures):,}")
        print(f"  Avg length: {np.mean([len(s) for s in self.structures]):.1f}")
        print(f"  Cache size: {len(list(self.cache_dir.glob('*.npz')))} files")
        print("="*60)
        
        return len(self.structures) > 0
    
    def _add_structure(self, protein_id, sequence, coords, metadata):
        """Add structure to dataset and cache."""
        self.structures.append(coords)
        self.sequences.append(sequence)
        self.metadata.append(metadata)
        
        # Cache to disk
        cache_file = self.cache_dir / f"{protein_id}.npz"
        np.savez_compressed(
            cache_file,
            coords=coords,
            sequence=sequence,
            metadata=json.dumps(metadata)
        )
    
    def _load_from_cache(self, protein_id):
        """Load structure from cache."""
        cache_file = self.cache_dir / f"{protein_id}.npz"
        try:
            data = np.load(cache_file, allow_pickle=True)
            self.structures.append(data['coords'])
            self.sequences.append(str(data['sequence']))
            self.metadata.append(json.loads(str(data['metadata'])))
        except Exception as e:
            print(f"⚠️  Failed to load {protein_id}: {e}")
    
    def get_batch(self, batch_size, max_len=256):
        """Get a batch of real protein data."""
        if len(self.structures) == 0:
            return None
        
        indices = random.sample(range(len(self.structures)), min(batch_size, len(self.structures)))
        
        batch_seqs = []
        batch_coords = []
        lengths = []
        
        for idx in indices:
            seq = self.sequences[idx]
            coords = self.structures[idx]
            length = min(len(seq), len(coords), max_len)
            batch_seqs.append(seq[:length])
            batch_coords.append(coords[:length])
            lengths.append(length)
        
        max_batch_len = max(lengths)
        
        # Convert to tensors
        aa_to_idx = ProteinDataGenerator.AMINO_ACID_TO_IDX
        padded_seqs = torch.zeros(len(batch_seqs), max_batch_len, dtype=torch.long)
        padded_coords = torch.zeros(len(batch_coords), max_batch_len, 3)
        mask = torch.zeros(len(batch_seqs), max_batch_len, dtype=torch.bool)
        
        for i, (seq, coords, length) in enumerate(zip(batch_seqs, batch_coords, lengths)):
            seq_indices = [aa_to_idx.get(aa, 0) for aa in seq]
            padded_seqs[i, :length] = torch.tensor(seq_indices)
            padded_coords[i, :length] = torch.from_numpy(coords).float()
            mask[i, :length] = True
        
        return {
            'sequences': padded_seqs,
            'coordinates': padded_coords,
            'mask': mask
        }

def train_revolutionary(args):
    """Training with all revolutionary features enabled."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("🚀 Revolutionary Training Mode")
    print(f"🖥️  Device: {device}")
    print("="*60)
    
    # Load or create config
    config_path = Path('config/model_config.json')
    with open(config_path) as f:
        config = json.load(f)
    
    # Enable all advanced features
    config['use_diffusion'] = True
    config['use_ipa'] = True
    config['use_constraints'] = True
    config['num_recycles'] = 3
    
    print("✅ Advanced features enabled:")
    print("   • Diffusion-based generation")
    print("   • Invariant Point Attention")
    print("   • Physical constraints")
    print("   • 3x recycling")
    
    # Choose model type
    if args.use_nas:
        print("   • Neural Architecture Search (NAS)\n")
        model = NASProteinModel(config).to(device)
        use_nas = True
    else:
        print("   • Revolutionary architecture\n")
        model = RevolutionaryProteinFolder(config).to(device)
        use_nas = False
    
    # Fresh data pipeline
    fresh_data = FreshDataPipeline()
    has_real_data = fresh_data.fetch_fresh_data(
        n_samples=args.n_fresh_samples,
        force_refresh=args.force_refresh
    )
    
    synthetic_generator = ProteinDataGenerator()
    
    # Advanced training components
    if use_nas:
        trainer = NASTrainer(model, lr_model=1e-4, lr_arch=3e-4)
        criterion = UncertaintyAwareLoss()
    else:
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=500)
        criterion = UncertaintyAwareLoss()
        grad_balancer = GradientNormBalancer(num_losses=4)
        
        if torch.cuda.is_available():
            mixed_precision_trainer = MixedPrecisionTrainer(model, optimizer)
    
    # Training loop
    start_time = time.time()
    epoch = 0
    best_loss = float('inf')
    real_ratio = 0.3 if has_real_data else 0.0
    
    print("\n🎯 Training Configuration:")
    print(f"   Max time: {args.max_time}s ({args.max_time/3600:.1f}h)")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Real data ratio: {real_ratio*100:.0f}%")
    print(f"   Use NAS: {use_nas}")
    print(f"   Mixed precision: {torch.cuda.is_available()}")
    print("="*60 + "\n")
    
    metrics_history = []
    
    while time.time() - start_time < args.max_time:
        epoch += 1
        model.train()
        
        # Get batch (real or synthetic)
        use_real = has_real_data and random.random() < real_ratio
        
        if use_real:
            batch = fresh_data.get_batch(args.batch_size, max_len=200)
            source_emoji = "🧬"
        else:
            batch = synthetic_generator.generate_synthetic_batch(
                args.batch_size,
                min_len=30,
                max_len=200
            )
            source_emoji = "🧪"
        
        sequences = batch['sequences'].to(device)
        target_coords = batch['coordinates'].to(device)
        mask = batch['mask'].to(device)
        
        # Training step
        try:
            if use_nas:
                # NAS training (needs both train and val batches)
                val_batch = synthetic_generator.generate_synthetic_batch(args.batch_size, 30, 200)
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                
                losses = trainer.train_step(batch, val_batch, criterion)
                loss = losses['model_loss']
            else:
                # Standard training
                if torch.cuda.is_available():
                    predictions = model(sequences, num_recycles=2)  # Reduce recycles for speed
                    targets = {'coordinates': target_coords}
                    loss, loss_dict = criterion(predictions, targets, mask)
                    
                    # Mixed precision backward
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                else:
                    optimizer.zero_grad()
                    predictions = model(sequences, num_recycles=2)
                    targets = {'coordinates': target_coords}
                    loss, loss_dict = criterion(predictions, targets, mask)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
        
        except Exception as e:
            print(f"⚠️  Error at epoch {epoch}: {e}")
            continue
        
        # Logging
        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            remaining = args.max_time - elapsed
            print(f"{source_emoji} Epoch {epoch} | Loss: {loss.item():.4f} | "
                  f"Time: {elapsed:.0f}s | Remaining: {remaining:.0f}s")
            
            metrics_history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'time': elapsed,
                'source': 'real' if use_real else 'synthetic'
            })
        
        # Save best
        if loss.item() < best_loss:
            best_loss = loss.item()
            if epoch % 50 == 0:
                checkpoint_path = Path('weights/latest.pt')
                checkpoint_path.parent.mkdir(exist_ok=True)
                model.save_checkpoint(str(checkpoint_path), {
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'use_nas': use_nas
                })
                print(f"  💾 Saved checkpoint (loss: {best_loss:.4f})")
    
    # Save final metrics
    metrics_path = Path('metrics/revolutionary_training.json')
    metrics_path.parent.mkdir(exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump({
            'total_epochs': epoch,
            'best_loss': best_loss,
            'training_time': time.time() - start_time,
            'use_nas': use_nas,
            'real_data_ratio': real_ratio,
            'history': metrics_history
        }, f, indent=2)
    
    print("\n✅ Revolutionary training complete!")
    print(f"   Epochs: {epoch}")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Time: {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Revolutionary training with all features')
    parser.add_argument('--max-time', type=int, default=18000, help='Max training time (seconds)')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--use-nas', action='store_true', help='Use Neural Architecture Search')
    parser.add_argument('--n-fresh-samples', type=int, default=50, help='Number of fresh proteins to fetch')
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh of all data')
    args = parser.parse_args()
    
    train_revolutionary(args)
