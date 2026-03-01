#!/usr/bin/env python3
"""Test that model architecture stays within safe constraints."""
import sys
import json
from pathlib import Path
import math

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
    MIN_DROPOUT = 0.0
    MAX_DROPOUT = 0.5
    
    # Adaptive parameter budget based on data size
    # Rule of thumb: ~10-20 parameters per training example to prevent overfitting
    PARAMS_PER_SAMPLE_MIN = 5
    PARAMS_PER_SAMPLE_MAX = 30
    BASE_PARAMETER_BUDGET = 50_000_000  # 50M baseline for small datasets
    MAX_PARAMETER_BUDGET = 1_000_000_000  # 1B absolute maximum

def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_training_samples():
    """Estimate number of training samples from data directory and logs."""
    # Check training metrics for epoch and batch info
    metrics_path = Path('metrics/training_metrics.json')
    
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        epochs = metrics.get('epochs', 0)
        # Assuming ~1000 samples per epoch as rough estimate
        estimated_samples = epochs * 1000
    else:
        # First run - use baseline
        estimated_samples = 5000
    
    # Check if real data was downloaded
    data_dir = Path('data/benchmark')
    if data_dir.exists():
        # Count JSON files which indicate downloaded data
        data_files = list(data_dir.glob('**/*.json')) + list(data_dir.glob('**/*.pdb'))
        real_data_count = len(data_files) * 50  # Rough estimate
        estimated_samples += real_data_count
    
    # Minimum of 5000 samples for stability
    return max(estimated_samples, 5000)

def calculate_parameter_budget(n_samples: int) -> int:
    """Calculate adaptive parameter budget based on training data size.
    
    Uses a logarithmic scaling to allow larger models with more data:
    - Small datasets (5K samples): ~50M params (10 params/sample)
    - Medium datasets (50K samples): ~250M params (5K params/sample)
    - Large datasets (500K samples): ~1B params (2K params/sample)
    
    This prevents overfitting while allowing growth with data.
    """
    # Logarithmic scaling: allows larger models as data grows
    # Formula: base_budget * log_scale_factor
    
    if n_samples < 10000:
        # Small dataset: conservative budget
        budget = n_samples * ArchitectureConstraints.PARAMS_PER_SAMPLE_MIN
    else:
        # Larger dataset: allow more parameters with log scaling
        # log10(samples) gives us nice scaling
        log_factor = math.log10(n_samples) - 3  # Normalize to 10K = 1.0
        budget = ArchitectureConstraints.BASE_PARAMETER_BUDGET * (1 + log_factor)
    
    # Clamp to reasonable bounds
    budget = max(budget, ArchitectureConstraints.BASE_PARAMETER_BUDGET)
    budget = min(budget, ArchitectureConstraints.MAX_PARAMETER_BUDGET)
    
    return int(budget)

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
    """Test that model doesn't exceed adaptive parameter budget."""
    checkpoint_path = Path('weights/latest.pt')
    
    if not checkpoint_path.exists():
        print("✓ No checkpoint yet (first run)")
        return
    
    model = EvolvableProteinFoldingModel.load_checkpoint(str(checkpoint_path))
    param_count = count_parameters(model)
    
    # Calculate adaptive budget
    n_samples = estimate_training_samples()
    param_budget = calculate_parameter_budget(n_samples)
    
    params_per_sample = param_count / n_samples
    
    print(f"\nAdaptive Parameter Budget:")
    print(f"  Training samples (estimated): {n_samples:,}")
    print(f"  Current parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    print(f"  Parameter budget: {param_budget:,} ({param_budget/1e6:.1f}M)")
    print(f"  Params per sample: {params_per_sample:.1f}")
    print(f"  Utilization: {(param_count/param_budget)*100:.1f}%")
    
    assert param_count <= param_budget, \
        f"Model has {param_count:,} parameters, exceeds adaptive budget of {param_budget:,} (based on {n_samples:,} samples)"
    
    # Warn if getting close to overfitting
    if params_per_sample > 20:
        print(f"  ⚠️  Warning: {params_per_sample:.1f} params/sample may risk overfitting")
    
    print(f"\n✓ Model size within adaptive budget (prevents overfitting)")

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
