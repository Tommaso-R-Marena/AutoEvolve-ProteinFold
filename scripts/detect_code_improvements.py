#!/usr/bin/env python3
"""Detect if code changes resulted in significant performance improvements."""
import json
import argparse
from pathlib import Path
import sys

def detect_improvements(threshold: float = 0.05):
    """Detect if performance improved beyond threshold.
    
    Args:
        threshold: Minimum improvement ratio (e.g., 0.05 = 5% improvement)
    
    Returns:
        bool: True if improvements exceed threshold
    """
    
    # Load performance history
    history_path = Path('metrics/performance_history.json')
    
    if not history_path.exists() or not history_path.stat().st_size:
        print("No performance history available yet")
        return False
    
    with open(history_path) as f:
        history = json.load(f)
    
    if len(history) < 2:
        print("Not enough history to detect improvements")
        return False
    
    # Compare recent performance to baseline
    # Use average of last 3 runs vs average of runs 4-6
    recent_runs = history[-3:]
    baseline_runs = history[-6:-3] if len(history) >= 6 else history[:-3]
    
    if not baseline_runs:
        print("Not enough baseline data")
        return False
    
    recent_avg = sum(r['rmsd'] for r in recent_runs) / len(recent_runs)
    baseline_avg = sum(r['rmsd'] for r in baseline_runs) / len(baseline_runs)
    
    # Calculate improvement (lower RMSD is better)
    if baseline_avg == 0:
        return False
    
    improvement_ratio = (baseline_avg - recent_avg) / baseline_avg
    
    print(f"\n{'='*60}")
    print("Code Improvement Detection:")
    print(f"{'='*60}")
    print(f"  Baseline RMSD (avg of {len(baseline_runs)} runs): {baseline_avg:.4f}")
    print(f"  Recent RMSD (avg of {len(recent_runs)} runs): {recent_avg:.4f}")
    print(f"  Improvement: {improvement_ratio*100:.2f}%")
    print(f"  Threshold: {threshold*100:.2f}%")
    
    if improvement_ratio >= threshold:
        print(f"\n  ✅ SIGNIFICANT IMPROVEMENT DETECTED!")
        print(f"     Performance gain of {improvement_ratio*100:.1f}% exceeds {threshold*100}% threshold")
        print(f"     Code changes will be committed.\n")
        print(f"{'='*60}\n")
        
        # Create flag file to signal workflow
        flag_file = Path('improvements_detected.flag')
        with open(flag_file, 'w') as f:
            f.write(json.dumps({
                'improvement_ratio': improvement_ratio,
                'baseline_rmsd': baseline_avg,
                'recent_rmsd': recent_avg,
                'threshold': threshold
            }, indent=2))
        
        return True
    else:
        print(f"\n  ℹ️  Improvement below threshold")
        print(f"     Only model weights will be committed (not code changes)\n")
        print(f"{'='*60}\n")
        
        # Remove flag file if exists
        flag_file = Path('improvements_detected.flag')
        if flag_file.exists():
            flag_file.unlink()
        
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.05,
                       help='Minimum improvement ratio to trigger code commit (default: 0.05 = 5%%)')
    args = parser.parse_args()
    
    improvements_found = detect_improvements(args.threshold)
    
    sys.exit(0 if improvements_found else 1)
