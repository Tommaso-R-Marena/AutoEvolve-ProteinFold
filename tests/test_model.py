#!/usr/bin/env python3
"""Unit tests for model functionality."""
import sys
from pathlib import Path
import torch
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from model.architecture import EvolvableProteinFoldingModel, EvoformerBlock, StructureModule
from model.data_generator import ProteinDataGenerator

class TestArchitecture:
    """Test model architecture components."""
    
    def test_model_forward_pass(self):
        """Test basic forward pass."""
        config = {
            'vocab_size': 20,
            'embedding_dim': 64,
            'pair_dim': 32,
            'n_features': 16,
            'n_heads': 4,
            'n_blocks': 2,
            'n_structure_blocks': 1,
            'dropout': 0.1
        }
        
        model = EvolvableProteinFoldingModel(config)
        model.eval()
        
        batch_size, seq_len = 2, 30
        x = torch.randint(0, 20, (batch_size, seq_len))
        
        with torch.no_grad():
            output = model(x)
        
        assert 'coordinates' in output
        assert 'angles' in output
        assert 'confidence' in output
        assert output['coordinates'].shape == (batch_size, seq_len, 3)
    
    def test_model_mutation(self):
        """Test architecture mutation."""
        config = {
            'vocab_size': 20,
            'embedding_dim': 64,
            'pair_dim': 32,
            'n_features': 16,
            'n_heads': 4,
            'n_blocks': 4,
            'n_structure_blocks': 1,
            'dropout': 0.1
        }
        
        model = EvolvableProteinFoldingModel(config)
        initial_blocks = len(model.evoformer_blocks)
        
        mutation_info = model.mutate_architecture(mutation_rate=1.0)
        
        assert model.generation > 0
        assert isinstance(mutation_info, dict)
    
    def test_checkpoint_save_load(self):
        """Test checkpoint save and load."""
        config = {
            'vocab_size': 20,
            'embedding_dim': 64,
            'pair_dim': 32,
            'n_features': 16,
            'n_heads': 4,
            'n_blocks': 2,
            'n_structure_blocks': 1,
            'dropout': 0.1
        }
        
        model = EvolvableProteinFoldingModel(config)
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            model.save_checkpoint(temp_path, metadata={'test': True})
            loaded_model = EvolvableProteinFoldingModel.load_checkpoint(temp_path)
            
            assert loaded_model.config == config
            assert loaded_model.generation == model.generation
        finally:
            Path(temp_path).unlink(missing_ok=True)

class TestDataGenerator:
    """Test data generation."""
    
    def test_synthetic_batch_generation(self):
        """Test synthetic data generation."""
        generator = ProteinDataGenerator()
        batch = generator.generate_synthetic_batch(batch_size=4, min_len=20, max_len=50)
        
        assert 'sequences' in batch
        assert 'coordinates' in batch
        assert 'lengths' in batch
        assert 'mask' in batch
        
        assert batch['sequences'].shape[0] == 4
        assert batch['coordinates'].shape[0] == 4
    
    def test_realistic_sequence(self):
        """Test that generated sequences are realistic."""
        generator = ProteinDataGenerator()
        sequence = generator._generate_realistic_sequence(100)
        
        assert len(sequence) == 100
        assert all(aa in generator.AMINO_ACIDS for aa in sequence)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
