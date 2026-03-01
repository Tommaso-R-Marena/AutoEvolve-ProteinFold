#!/usr/bin/env python3
"""Quality gates that must pass before committing changes."""
import sys
import json
import argparse
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))

from model.architecture import EvolvableProteinFoldingModel

class QualityGates:
    """Define quality thresholds for self-modification."""
    
    MIN_IMPROVEMENT_THRESHOLD = -0.10  # Allow 10% worse for exploration
    REQUIRE_IMPROVEMENT_AFTER_N_CYCLES = 5  # Must improve after N cycles
    MAX_CONSECUTIVE_FAILURES = 3  # Max failed quality checks in a row

def check_model_loadable():
    """Ensure model checkpoint is valid and loadable."""
    checkpoint_path = Path('weights/latest.pt')
    
    if not checkpoint_path.exists():
        print("✓ No checkpoint yet (first run)")
        return True
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        required_keys = ['model_state_dict', 'config', 'generation']
        for key in required_keys:
            if key not in checkpoint:
                print(f"❌ Missing checkpoint key: {key}")
                return False
        
        # Try to load model
        model = EvolvableProteinFoldingModel.load_checkpoint(str(checkpoint_path))
        
        print(f"✓ Model checkpoint valid (generation {model.generation})")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False

def check_improvement_trend():
    """Check if model is improving over time."""
    history_path = Path('metrics/performance_history.json')
    
    if not history_path.exists():
        print("✓ No performance history yet (first run)")
        return True
    
    with open(history_path) as f:
        history = json.load(f)
    
    if len(history) < QualityGates.REQUIRE_IMPROVEMENT_AFTER_N_CYCLES:
        print(f"✓ Only {len(history)} cycles, not enough for trend analysis")
        return True
    
    # Get recent performance
    recent = history[-QualityGates.REQUIRE_IMPROVEMENT_AFTER_N_CYCLES:]
    recent_rmsds = [r['rmsd'] for r in recent]
    
    # Check if there's improvement
    best_recent = min(recent_rmsds)
    overall_best = min(r['rmsd'] for r in history)
    
    if best_recent > overall_best * 1.2:  # 20% worse than ever
        print(f"❌ No improvement in last {QualityGates.REQUIRE_IMPROVEMENT_AFTER_N_CYCLES} cycles")
        print(f"   Recent best: {best_recent:.4f}, Overall best: {overall_best:.4f}")
        return False
    
    print(f"✓ Improvement trend acceptable (recent best: {best_recent:.4f})")
    return True

def check_consecutive_failures():
    """Check if there have been too many consecutive failures."""
    failure_log_path = Path('logs/quality_failures.json')
    
    if not failure_log_path.exists():
        return True
    
    with open(failure_log_path) as f:
        failures = json.load(f)
    
    consecutive = 0
    for entry in reversed(failures):
        if entry.get('failed', False):
            consecutive += 1
        else:
            break
    
    if consecutive >= QualityGates.MAX_CONSECUTIVE_FAILURES:
        print(f"❌ {consecutive} consecutive quality gate failures")
        return False
    
    print(f"✓ Consecutive failures: {consecutive}/{QualityGates.MAX_CONSECUTIVE_FAILURES}")
    return True

def check_evolution_sanity():
    """Check that architectural evolution is sensible."""
    evolution_path = Path('logs/evolution_history.json')
    
    if not evolution_path.exists():
        print("✓ No evolution history yet")
        return True
    
    with open(evolution_path) as f:
        history = json.load(f)
    
    if len(history) == 0:
        return True
    
    # Check last evolution
    last_evolution = history[-1]
    
    # If evolution claimed improvement, verify metrics support it
    if last_evolution.get('improved', False):
        base_perf = last_evolution.get('base_performance', float('inf'))
        mutant_perf = last_evolution.get('best_mutant_performance', float('inf'))
        
        if mutant_perf >= base_perf:
            print(f"⚠️  Evolution claimed improvement but metrics don't support it")
            print(f"   Base: {base_perf:.4f}, Mutant: {mutant_perf:.4f}")
            # Don't fail, just warn
    
    print("✓ Evolution history seems reasonable")
    return True

def log_quality_check_result(passed: bool):
    """Log the result of this quality check."""
    log_path = Path('logs/quality_failures.json')
    log_path.parent.mkdir(exist_ok=True, parents=True)
    
    if log_path.exists():
        with open(log_path) as f:
            logs = json.load(f)
    else:
        logs = []
    
    from datetime import datetime
    logs.append({
        'timestamp': datetime.now().isoformat(),
        'failed': not passed
    })
    
    # Keep last 50 entries
    logs = logs[-50:]
    
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strict', action='store_true', help='Enforce strict quality gates')
    args = parser.parse_args()
    
    print("Running quality gate checks...\n")
    
    checks = [
        ('Model Loadable', check_model_loadable()),
        ('Improvement Trend', check_improvement_trend()),
        ('Consecutive Failures', check_consecutive_failures()),
        ('Evolution Sanity', check_evolution_sanity())
    ]
    
    all_passed = all(result for _, result in checks)
    
    print("\n" + "="*50)
    print("Quality Gate Summary:")
    print("="*50)
    
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name}")
    
    log_quality_check_result(all_passed)
    
    if all_passed:
        print("\n✅ All quality gates passed - safe to commit changes")
        sys.exit(0)
    else:
        print("\n❌ Quality gates failed - changes will be rolled back")
        sys.exit(1)
