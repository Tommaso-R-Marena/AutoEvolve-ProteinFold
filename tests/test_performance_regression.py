#!/usr/bin/env python3
"""Test for performance regression - ensure model doesn't get worse."""
import sys
import json
import argparse
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

class PerformanceThresholds:
    """Performance thresholds that must be maintained."""
    
    MAX_LOSS_DEGRADATION = 0.15  # Allow 15% loss increase
    MIN_CONFIDENCE = 0.3  # Minimum average confidence
    MAX_LOSS_ABSOLUTE = 50.0  # Absolute maximum loss value

def load_performance_history():
    """Load historical performance metrics."""
    metrics_path = Path('metrics/performance_history.json')
    
    if not metrics_path.exists():
        return []
    
    with open(metrics_path) as f:
        return json.load(f)

def save_performance_record(record):
    """Save current performance to history."""
    history = load_performance_history()
    history.append(record)
    
    # Keep last 100 records
    history = history[-100:]
    
    metrics_path = Path('metrics/performance_history.json')
    metrics_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(metrics_path, 'w') as f:
        json.dump(history, f, indent=2)

def test_performance_regression(checkpoint='latest'):
    """Test that performance hasn't regressed significantly."""
    
    # Load current metrics
    current_metrics_path = Path('metrics/evaluation_results.json')
    
    if not current_metrics_path.exists():
        print("✓ No evaluation results yet (first run)")
        return
    
    with open(current_metrics_path) as f:
        current_metrics = json.load(f)
    
    current_rmsd = current_metrics.get('avg_rmsd', float('inf'))
    
    # Check absolute threshold
    assert current_rmsd < PerformanceThresholds.MAX_LOSS_ABSOLUTE, \
        f"RMSD {current_rmsd:.4f} exceeds absolute maximum {PerformanceThresholds.MAX_LOSS_ABSOLUTE}"
    
    # Load performance history
    history = load_performance_history()
    
    if len(history) == 0:
        print(f"✓ First performance record: RMSD = {current_rmsd:.4f}")
        save_performance_record({
            'checkpoint': checkpoint,
            'rmsd': current_rmsd,
            'generation': current_metrics.get('generation', 0)
        })
        return
    
    # Get best historical performance
    best_rmsd = min(record['rmsd'] for record in history)
    
    # Calculate degradation
    degradation = (current_rmsd - best_rmsd) / best_rmsd
    
    # Allow some degradation (exploration) but not too much
    assert degradation < PerformanceThresholds.MAX_LOSS_DEGRADATION, \
        f"Performance degraded by {degradation*100:.1f}% (current: {current_rmsd:.4f}, best: {best_rmsd:.4f}). Max allowed: {PerformanceThresholds.MAX_LOSS_DEGRADATION*100:.1f}%"
    
    if current_rmsd <= best_rmsd:
        print(f"✓ NEW BEST performance: RMSD = {current_rmsd:.4f} (previous best: {best_rmsd:.4f})")
    else:
        print(f"✓ Performance acceptable: RMSD = {current_rmsd:.4f} (best: {best_rmsd:.4f}, degradation: {degradation*100:.1f}%)")
    
    # Save current performance
    save_performance_record({
        'checkpoint': checkpoint,
        'rmsd': current_rmsd,
        'generation': current_metrics.get('generation', 0)
    })

def test_training_metrics():
    """Test that training metrics are reasonable."""
    training_metrics_path = Path('metrics/training_metrics.json')
    
    if not training_metrics_path.exists():
        print("✓ No training metrics yet (first run)")
        return
    
    with open(training_metrics_path) as f:
        metrics = json.load(f)
    
    final_loss = metrics.get('final_loss', float('inf'))
    best_loss = metrics.get('best_loss', float('inf'))
    
    # Check for NaN or inf
    assert np.isfinite(final_loss), "Training produced NaN or Inf loss"
    assert np.isfinite(best_loss), "Best loss is NaN or Inf"
    
    # Check loss is positive
    assert final_loss >= 0, f"Invalid negative loss: {final_loss}"
    assert best_loss >= 0, f"Invalid negative best loss: {best_loss}"
    
    print(f"✓ Training metrics valid: final_loss={final_loss:.4f}, best_loss={best_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='latest')
    args = parser.parse_args()
    
    print("Testing for performance regression...\n")
    
    try:
        test_performance_regression(args.checkpoint)
        test_training_metrics()
        
        print("\n✅ Performance regression tests passed!")
        sys.exit(0)
        
    except AssertionError as e:
        print(f"\n❌ Performance regression detected: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
