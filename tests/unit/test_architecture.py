"""Unit tests for model architecture."""
import pytest
import torch
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from model.architecture import EvolvableProteinFoldingModel, EvoformerBlock, StructureModule


@pytest.fixture
def model_config():
    """Load model configuration."""
    config_path = Path(__file__).parent.parent.parent / 'config' / 'model_config.json'
    with open(config_path) as f:
        return json.load(f)


@pytest.fixture
def model(model_config):
    """Create model instance."""
    return EvolvableProteinFoldingModel(model_config)


class TestModelCreation:
    def test_model_initialization(self, model_config):
        """Test model can be created."""
        model = EvolvableProteinFoldingModel(model_config)
        assert model is not None
        assert model.config == model_config
    
    def test_model_parameters(self, model):
        """Test model has trainable parameters."""
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)
    
    def test_embedding_dimensions(self, model, model_config):
        """Test embedding layer has correct dimensions."""
        assert model.amino_acid_embedding.embedding_dim == model_config['embedding_dim']
        assert model.amino_acid_embedding.num_embeddings == model_config['vocab_size']


class TestForwardPass:
    def test_forward_pass_shape(self, model):
        """Test forward pass produces correct output shapes."""
        batch_size, seq_len = 2, 50
        sequences = torch.randint(0, 20, (batch_size, seq_len))
        
        outputs = model(sequences)
        
        assert 'coordinates' in outputs
        assert 'angles' in outputs
        assert 'confidence' in outputs
        assert outputs['coordinates'].shape == (batch_size, seq_len, 3)
        assert outputs['angles'].shape == (batch_size, seq_len, 3)
        assert outputs['confidence'].shape == (batch_size, seq_len)
    
    def test_forward_pass_no_nan(self, model):
        """Test forward pass doesn't produce NaN values."""
        sequences = torch.randint(0, 20, (1, 30))
        outputs = model(sequences)
        
        assert not torch.isnan(outputs['coordinates']).any()
        assert not torch.isnan(outputs['angles']).any()
        assert not torch.isnan(outputs['confidence']).any()
    
    def test_confidence_in_valid_range(self, model):
        """Test confidence scores are between 0 and 1."""
        sequences = torch.randint(0, 20, (1, 30))
        outputs = model(sequences)
        
        confidence = outputs['confidence']
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()
    
    def test_different_sequence_lengths(self, model):
        """Test model handles variable sequence lengths."""
        for seq_len in [10, 50, 100, 200]:
            sequences = torch.randint(0, 20, (1, seq_len))
            outputs = model(sequences)
            assert outputs['coordinates'].shape[1] == seq_len


class TestCheckpointSaveLoad:
    def test_save_checkpoint(self, model, tmp_path):
        """Test model can be saved."""
        checkpoint_path = tmp_path / 'test_checkpoint.pt'
        model.save_checkpoint(str(checkpoint_path))
        assert checkpoint_path.exists()
    
    def test_load_checkpoint(self, model, tmp_path):
        """Test model can be loaded."""
        checkpoint_path = tmp_path / 'test_checkpoint.pt'
        model.save_checkpoint(str(checkpoint_path))
        
        loaded_model = EvolvableProteinFoldingModel.load_checkpoint(str(checkpoint_path))
        assert loaded_model.config == model.config
    
    def test_checkpoint_preserves_weights(self, model, tmp_path):
        """Test saved/loaded models have same weights."""
        checkpoint_path = tmp_path / 'test_checkpoint.pt'
        
        # Get original weights
        original_weights = {name: param.clone() for name, param in model.named_parameters()}
        
        # Save and load
        model.save_checkpoint(str(checkpoint_path))
        loaded_model = EvolvableProteinFoldingModel.load_checkpoint(str(checkpoint_path))
        
        # Compare weights
        for name, param in loaded_model.named_parameters():
            assert torch.allclose(param, original_weights[name])


class TestEvoformerBlock:
    def test_evoformer_forward(self, model_config):
        """Test EvoformerBlock forward pass."""
        block = EvoformerBlock(
            model_config['embedding_dim'],
            model_config['pair_dim'],
            model_config['n_heads'],
            model_config['dropout']
        )
        
        batch_size, seq_len = 2, 30
        seq_feat = torch.randn(batch_size, seq_len, model_config['embedding_dim'])
        pair_feat = torch.randn(batch_size, seq_len, seq_len, model_config['pair_dim'])
        
        seq_out, pair_out = block(seq_feat, pair_feat)
        
        assert seq_out.shape == seq_feat.shape
        assert pair_out.shape == pair_feat.shape
        assert not torch.isnan(seq_out).any()
        assert not torch.isnan(pair_out).any()


class TestStructureModule:
    def test_structure_module_forward(self, model_config):
        """Test StructureModule forward pass."""
        structure_module = StructureModule(
            model_config['embedding_dim'],
            model_config['pair_dim'],
            model_config['n_structure_blocks']
        )
        
        batch_size, seq_len = 2, 40
        seq_feat = torch.randn(batch_size, seq_len, model_config['embedding_dim'])
        pair_feat = torch.randn(batch_size, seq_len, seq_len, model_config['pair_dim'])
        
        coords, angles = structure_module(seq_feat, pair_feat)
        
        assert coords.shape == (batch_size, seq_len, 3)
        assert angles.shape == (batch_size, seq_len, 3)
        assert not torch.isnan(coords).any()
        assert not torch.isnan(angles).any()


class TestEvolution:
    def test_mutation(self, model):
        """Test architecture mutation."""
        original_blocks = len(model.evoformer_blocks)
        mutation_info = model.mutate_architecture(mutation_rate=1.0)
        
        assert 'mutations' in mutation_info
        assert 'generation' in mutation_info
        assert model.generation == 1
    
    def test_generation_counter(self, model):
        """Test generation counter increments."""
        assert model.generation == 0
        model.mutate_architecture()
        assert model.generation == 1
        model.mutate_architecture()
        assert model.generation == 2
