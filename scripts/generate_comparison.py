#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from datetime import datetime

def generate_comparison(args):
    """Generate markdown comparison report."""
    
    results_path = Path('metrics/benchmark_results.json')
    
    if not results_path.exists():
        print("No benchmark results found")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    # Generate markdown report
    report = f"""# Benchmark Comparison Report
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

## Overview

This report compares AutoEvolve-ProteinFold against state-of-the-art protein folding models.

"""
    
    # Check if comparison exists
    if 'comparison' in results:
        comparison = results['comparison']
        
        report += "## Model Performance Comparison\n\n"
        report += "| Model | Avg Confidence | Notes |\n"
        report += "|-------|---------------|-------|\n"
        
        for model_name, model_data in comparison.items():
            confidence = model_data.get('avg_confidence', 'N/A')
            if isinstance(confidence, float):
                confidence = f"{confidence:.3f}"
            
            notes = ""
            if model_name == 'our_model':
                gen = model_data.get('generation', 'N/A')
                notes = f"Generation {gen}"
            
            report += f"| {model_name} | {confidence} | {notes} |\n"
    
    # Dataset results
    report += "\n## Dataset Results\n\n"
    
    for dataset_name, dataset_results in results.items():
        if dataset_name == 'comparison':
            continue
        
        if isinstance(dataset_results, dict):
            report += f"### {dataset_name}\n\n"
            report += f"- Targets evaluated: {dataset_results.get('n_targets', 'N/A')}\n"
            report += f"- Successful predictions: {dataset_results.get('n_successful', 'N/A')}\n"
            report += f"- Average confidence: {dataset_results.get('avg_confidence', 'N/A'):.3f}\n\n"
    
    report += "\n## Interpretation\n\n"
    report += "- **Confidence scores** reflect model certainty in predictions (0-1 scale)\n"
    report += "- Higher scores indicate more reliable structural predictions\n"
    report += "- Model improves continuously through evolutionary training\n"
    
    report += "\n---\n"
    report += "*Note: This is an evolving research project. Performance metrics will improve over time.*\n"
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Comparison report saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    generate_comparison(args)
