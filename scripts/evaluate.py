#!/usr/bin/env python3
import torch
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from model.architecture import EvolvableProteinFoldingModel
from model.data_generator import ProteinDataGenerator

def evaluate(args):
    checkpoint_path = Path(f'weights/{args.checkpoint}.pt')
    
    if not checkpoint_path.exists():
        print(f"Checkpoint {checkpoint_path} not found")
        return
    
    print(f"Loading model from {checkpoint_path}...")
    model = EvolvableProteinFoldingModel.load_checkpoint(str(checkpoint_path))
    model.eval()
    
    data_generator = ProteinDataGenerator()
    
    print("Evaluating model performance...")
    total_rmsd = 0
    n_samples = 50
    
    with torch.no_grad():
        for i in range(n_samples):
            batch = data_generator.generate_synthetic_batch(1)
            sequences = batch['sequences']
            target_coords = batch['coordinates']
            mask = batch['mask']
            
            predictions = model(sequences)
            
            # Calculate RMSD
            coord_diff = (predictions['coordinates'] - target_coords) ** 2
            rmsd = torch.sqrt((coord_diff * mask.unsqueeze(-1)).sum() / mask.sum())
            total_rmsd += rmsd.item()
    
    avg_rmsd = total_rmsd / n_samples
    print(f"Average RMSD: {avg_rmsd:.4f} Å")
    
    # Save evaluation results
    results = {
        'checkpoint': args.checkpoint,
        'generation': model.generation,
        'avg_rmsd': avg_rmsd,
        'n_samples': n_samples
    }
    
    results_path = Path('metrics/evaluation_results.json')
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='latest')
    args = parser.parse_args()
    
    evaluate(args)
