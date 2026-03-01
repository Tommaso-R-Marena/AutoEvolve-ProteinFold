#!/usr/bin/env python3
"""Test that model architecture stays within safe constraints."""
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from model.architecture import EvolvableProteinFoldingModel

class ArchitectureConstraints:
    """Define safe boundaries for model architecture."""
    
    MIN_BLOCKS = 2
    MAX_BLOCKS = 24
    MIN_EMBEDDING_DIM = 64
    MAX_EMBEDDING_DIM = 1024
    MIN_HEADS = 2
    MAX_HEADS = 16
    MAX_PARAMETERS = 500_000_000  # 500M parameters max
    MIN_DROPOUT = 0.0
    MAX_DROPOUT = 0.5

def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_config_constraints():
    """Test that config file adheres to constraints."""
    config_path = Path('config/model_config.json')
    
    if not config_path.exists():
        print("✓ No config file yet (first run)")
        return
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Check block count
    n_blocks = config.get('n_blocks', 0)
    assert ArchitectureConstraints.MIN_BLOCKS <= n_blocks <= ArchitectureConstraints.MAX_BLOCKS, \
        f"Block count {n_blocks} outside safe range [{ArchitectureConstraints.MIN_BLOCKS}, {ArchitectureConstraints.MAX_BLOCKS}]"
    
    # Check embedding dimension
    embed_dim = config.get('embedding_dim', 0)
    assert ArchitectureConstraints.MIN_EMBEDDING_DIM <= embed_dim <= ArchitectureConstraints.MAX_EMBEDDING_DIM, \
        f"Embedding dim {embed_dim} outside safe range"
    
    # Check attention heads
    n_heads = config.get('n_heads', 0)
    assert ArchitectureConstraints.MIN_HEADS <= n_heads <= ArchitectureConstraints.MAX_HEADS, \
        f"Attention heads {n_heads} outside safe range"
    
    # Check dropout
    dropout = config.get('dropout', 0)
    assert ArchitectureConstraints.MIN_DROPOUT <= dropout <= ArchitectureConstraints.MAX_DROPOUT, \
        f"Dropout {dropout} outside safe range"
    
    # Check embedding dim divisible by heads
    assert embed_dim % n_heads == 0, \
        f"Embedding dim {embed_dim} must be divisible by n_heads {n_heads}"
    
    print(f"✓ Config constraints satisfied: {n_blocks} blocks, {embed_dim}D, {n_heads} heads")

def test_model_size():
    """Test that model doesn't exceed parameter budget."""
    checkpoint_path = Path('weights/latest.pt')
    
    if not checkpoint_path.exists():
        print("✓ No checkpoint yet (first run)")
        return
    
    model = EvolvableProteinFoldingModel.load_checkpoint(str(checkpoint_path))
    param_count = count_parameters(model)
    
    assert param_count <= ArchitectureConstraints.MAX_PARAMETERS, \
        f"Model has {param_count:,} parameters, exceeds limit of {ArchitectureConstraints.MAX_PARAMETERS:,}"
    
    print(f"✓ Model size OK: {param_count:,} parameters ({param_count/1e6:.1f}M)")

def test_architecture_validity():
    """Test that model can be instantiated and forward pass works."""
    checkpoint_path = Path('weights/latest.pt')
    
    if not checkpoint_path.exists():
        print("✓ No checkpoint yet (first run)")
        return
    
    import torch
    
    model = EvolvableProteinFoldingModel.load_checkpoint(str(checkpoint_path))
    model.eval()
    
    # Test forward pass with dummy input
    batch_size = 2
    seq_len = 50
    dummy_input = torch.randint(0, 20, (batch_size, seq_len))
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        
        # Verify output structure
        required_keys = ['coordinates', 'angles', 'confidence']
        for key in required_keys:
            assert key in output, f"Missing output key: {key}"
        
        # Verify output shapes
        assert output['coordinates'].shape == (batch_size, seq_len, 3), "Invalid coordinate shape"
        assert output['angles'].shape == (batch_size, seq_len, 3), "Invalid angle shape"
        assert output['confidence'].shape == (batch_size, seq_len), "Invalid confidence shape"
        
        print("✓ Architecture valid: forward pass successful")
        
    except Exception as e:
        raise AssertionError(f"Model forward pass failed: {e}")

if __name__ == '__main__':
    print("Testing architecture constraints...\n")
    
    try:
        test_config_constraints()
        test_model_size()
        test_architecture_validity()
        
        print("\n✅ All architecture constraint tests passed!")
        sys.exit(0)
        
    except AssertionError as e:
        print(f"\n❌ Architecture constraint violation: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
