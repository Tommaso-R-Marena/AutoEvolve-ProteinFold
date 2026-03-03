"""Memory usage tests to ensure model fits in GitHub Actions limits."""
import pytest
import torch
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from model.architecture import EvolvableProteinFoldingModel

def get_test_config():
    """Get standard test configuration."""
    return {
        'vocab_size': 20,
        'embedding_dim': 128,
        'pair_dim': 64,
        'n_heads': 8,
        'n_blocks': 2,
        'n_structure_blocks': 3,
        'dropout': 0.1,
        'max_sequence_length': 256
    }

@pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
def test_model_memory_footprint():
    """Test that model fits in memory limits."""
    if not PSUTIL_AVAILABLE:
        pytest.skip("psutil not available")
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create model
    config = get_test_config()
    model = EvolvableProteinFoldingModel(config)
    
    # Check memory after model creation
    model_memory = process.memory_info().rss / 1024 / 1024
    memory_used = model_memory - initial_memory
    
    print(f"\nMemory usage:")
    print(f"  Initial: {initial_memory:.1f} MB")
    print(f"  After model: {model_memory:.1f} MB")
    print(f"  Model size: {memory_used:.1f} MB")
    
    # GitHub Actions has ~7GB available, leave plenty of room
    assert memory_used < 500, f"Model uses too much memory: {memory_used:.1f} MB"

@pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
def test_training_memory_footprint():
    """Test memory during training."""
    if not PSUTIL_AVAILABLE:
        pytest.skip("psutil not available")
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    config = get_test_config()
    model = EvolvableProteinFoldingModel(config)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Simulate training step
    batch = torch.randint(0, 20, (2, 100))
    outputs = model(batch)
    loss = outputs['coordinates'].sum()
    loss.backward()
    optimizer.step()
    
    training_memory = process.memory_info().rss / 1024 / 1024
    total_used = training_memory - initial_memory
    
    print(f"\nTraining memory:")
    print(f"  Total used: {total_used:.1f} MB")
    
    # Should fit comfortably in 7GB with room for data
    assert total_used < 1000, f"Training uses too much memory: {total_used:.1f} MB"

def test_model_parameters_count():
    """Test that model has reasonable parameter count."""
    config = get_test_config()
    model = EvolvableProteinFoldingModel(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter counts:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Size estimate: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    # Should be under 10M parameters for efficiency
    assert total_params < 10_000_000, f"Too many parameters: {total_params:,}"
    assert trainable_params == total_params, "All parameters should be trainable"
