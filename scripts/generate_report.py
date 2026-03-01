#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from datetime import datetime

def generate_report(args):
    # Load metrics
    metrics_path = Path('metrics/training_metrics.json')
    if not metrics_path.exists():
        print("No metrics found")
        return
    
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # Load evolution history
    evolution_path = Path('logs/evolution_history.json')
    evolution_history = []
    if evolution_path.exists():
        with open(evolution_path) as f:
            evolution_history = json.load(f)
    
    # Generate report
    report = f"""# Training Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Current Status
- **Generation**: {metrics.get('generation', 0)}
- **Training Epochs**: {metrics.get('epochs', 0)}
- **Final Loss**: {metrics.get('final_loss', 0):.4f}
- **Best Loss**: {metrics.get('best_loss', 0):.4f}
- **Training Time**: {metrics.get('training_time', 0):.1f}s

## Evolution History
{len(evolution_history)} architectural evolutions completed

## Next Steps
- Continue training
- Benchmark against SOTA
- Fetch real protein data from UniProt
"""
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    generate_report(args)
