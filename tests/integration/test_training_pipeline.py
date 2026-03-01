"""Integration tests for training pipeline."""
import pytest
import torch
import json
import tempfile
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from model.architecture import EvolvableProteinFoldingModel
from model.data_generator import ProteinDataGenerator
from scripts.train_cycle import ProteinFoldingLoss, TrainingState


class TestTrainingCycle:
    def test_single_training_step(self):
        """Test one complete training step."""
        # Setup
        config_path = Path(__file__).parent.parent.parent / 'config' / 'model_config.json'
        with open(config_path) as f:
            config = json.load(f)
        
        model = EvolvableProteinFoldingModel(config)
        generator = ProteinDataGenerator()
        criterion = ProteinFoldingLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Generate batch
        batch = generator.generate_synthetic_batch(2, min_len=30, max_len=50)
        sequences = batch['sequences']
        target_coords = batch['coordinates']
        mask = batch['mask']
        target_angles = torch.randn_like(target_coords)
        
        # Forward pass
        predictions = model(sequences)
        
        # Compute loss
        targets = {'coordinates': target_coords, 'angles': target_angles}
        loss, loss_dict = criterion(predictions, targets, mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verify
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert all(not torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    
    def test_loss_decreases(self):
        """Test loss decreases over multiple steps."""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'model_config.json'
        with open(config_path) as f:
            config = json.load(f)
        
        model = EvolvableProteinFoldingModel(config)
        generator = ProteinDataGenerator()
        criterion = ProteinFoldingLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Same batch for multiple iterations
        batch = generator.generate_synthetic_batch(2, min_len=30, max_len=30)
        sequences = batch['sequences']
        target_coords = batch['coordinates']
        mask = batch['mask']
        target_angles = torch.randn_like(target_coords)
        targets = {'coordinates': target_coords, 'angles': target_angles}
        
        losses = []
        for _ in range(10):
            predictions = model(sequences)
            loss, _ = criterion(predictions, targets, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Loss should generally decrease
        assert losses[-1] < losses[0]


class TestTrainingState:
    def test_state_save_load(self):
        """Test training state can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_manager = TrainingState(tmpdir)
            
            # Save state
            state_manager.save_state(
                epoch=100,
                optimizer_state={'dummy': 'state'},
                scheduler_state={'dummy': 'state'},
                total_samples=1000,
                best_loss=5.0,
                rng_state={'torch': torch.get_rng_state(), 'numpy': None}
            )
            
            # Load state
            loaded = state_manager.load_state()
            
            assert loaded is not None
            assert loaded['epoch'] == 100
            assert loaded['total_samples'] == 1000
            assert loaded['best_loss'] == 5.0
