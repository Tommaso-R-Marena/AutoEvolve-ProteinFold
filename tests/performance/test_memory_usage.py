"""Performance and memory usage tests."""
import torch
import json
import psutil
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from model.architecture import EvolvableProteinFoldingModel


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_memory_under_limit():
    """Test model stays under 7GB GitHub Actions limit."""
    config_path = Path(__file__).parent.parent.parent / 'config' / 'model_config.json'
    with open(config_path) as f:
        config = json.load(f)
    
    # Measure initial memory
    initial_mem = get_memory_usage_mb()
    print(f"\n📈 Initial memory: {initial_mem:.1f} MB")
    
    # Create model
    model = EvolvableProteinFoldingModel(config)
    after_model_mem = get_memory_usage_mb()
    print(f"🎯 Model creation: +{after_model_mem - initial_mem:.1f} MB (total: {after_model_mem:.1f} MB)")
    
    # Forward pass with largest expected batch
    batch_size = 4
    seq_len = 200
    sequences = torch.randint(0, 20, (batch_size, seq_len))
    
    outputs = model(sequences)
    after_forward_mem = get_memory_usage_mb()
    print(f"🚀 Forward pass: +{after_forward_mem - after_model_mem:.1f} MB (total: {after_forward_mem:.1f} MB)")
    
    # Backward pass simulation
    loss = outputs['coordinates'].sum()
    loss.backward()
    after_backward_mem = get_memory_usage_mb()
    print(f"⏪ Backward pass: +{after_backward_mem - after_forward_mem:.1f} MB (total: {after_backward_mem:.1f} MB)")
    
    # Check limit
    GITHUB_ACTIONS_LIMIT_MB = 7000  # 7 GB
    print(f"\n🎯 Peak memory: {after_backward_mem:.1f} MB / {GITHUB_ACTIONS_LIMIT_MB} MB")
    print(f"🔴 Headroom: {GITHUB_ACTIONS_LIMIT_MB - after_backward_mem:.1f} MB ({(1 - after_backward_mem/GITHUB_ACTIONS_LIMIT_MB)*100:.1f}%)")
    
    assert after_backward_mem < GITHUB_ACTIONS_LIMIT_MB, \
        f"Memory usage {after_backward_mem:.1f} MB exceeds GitHub Actions limit of {GITHUB_ACTIONS_LIMIT_MB} MB"
    
    print("\n✅ Memory usage test PASSED!")


if __name__ == '__main__':
    test_memory_under_limit()
