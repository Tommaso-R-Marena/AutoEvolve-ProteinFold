#!/usr/bin/env python3
import torch
import argparse
import json
import sys
from pathlib import Path
import copy
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from model.architecture import EvolvableProteinFoldingModel
from model.data_generator import ProteinDataGenerator

def evaluate_model(model, data_generator, n_batches=10):
    """Evaluate model performance on synthetic data."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for _ in range(n_batches):
            batch = data_generator.generate_synthetic_batch(4)
            sequences = batch['sequences']
            target_coords = batch['coordinates']
            mask = batch['mask']
            
            predictions = model(sequences)
            
            # Simple RMSD loss
            coord_diff = (predictions['coordinates'] - target_coords) ** 2
            rmsd = torch.sqrt((coord_diff * mask.unsqueeze(-1)).sum() / mask.sum())
            total_loss += rmsd.item()
    
    return total_loss / n_batches

def evolve_architecture(args):
    """Evolutionary architecture search."""
    print("Starting evolutionary architecture search...")
    
    checkpoint_path = Path('weights/latest.pt')
    
    if not checkpoint_path.exists():
        print("No checkpoint found. Run training first.")
        return
    
    # Load base model
    base_model = EvolvableProteinFoldingModel.load_checkpoint(str(checkpoint_path))
    data_generator = ProteinDataGenerator()
    
    # Evaluate base model
    base_performance = evaluate_model(base_model, data_generator)
    print(f"Base model performance: {base_performance:.4f}")
    
    # Create population of mutated models
    population = []
    for i in range(args.population_size):
        mutant = copy.deepcopy(base_model)
        mutation_info = mutant.mutate_architecture(mutation_rate=0.2)
        performance = evaluate_model(mutant, data_generator)
        
        population.append({
            'model': mutant,
            'performance': performance,
            'mutations': mutation_info
        })
        
        print(f"Mutant {i+1}: {performance:.4f} - {mutation_info['mutations']}")
    
    # Select best performing mutant
    population.sort(key=lambda x: x['performance'])
    best_mutant = population[0]
    
    print(f"\nBest mutant performance: {best_mutant['performance']:.4f}")
    
    # Save if better than base
    if best_mutant['performance'] < base_performance:
        print("Saving improved architecture...")
        best_mutant['model'].save_checkpoint(
            str(checkpoint_path),
            metadata={
                'evolution_generation': best_mutant['model'].generation,
                'mutations': best_mutant['mutations'],
                'performance': best_mutant['performance']
            }
        )
        
        # Update config
        config_path = Path('config/model_config.json')
        with open(config_path, 'w') as f:
            json.dump(best_mutant['model'].config, f, indent=2)
        
        print("Architecture evolved successfully!")
    else:
        print("No improvement found. Keeping base architecture.")
    
    # Log evolution history
    evolution_log_path = Path('logs/evolution_history.json')
    evolution_log_path.parent.mkdir(exist_ok=True)
    
    evolution_entry = {
        'generation': base_model.generation,
        'base_performance': base_performance,
        'best_mutant_performance': best_mutant['performance'],
        'mutations': best_mutant['mutations'],
        'improved': best_mutant['performance'] < base_performance
    }
    
    # Append to history
    if evolution_log_path.exists():
        with open(evolution_log_path) as f:
            history = json.load(f)
    else:
        history = []
    
    history.append(evolution_entry)
    
    with open(evolution_log_path, 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--population-size', type=int, default=5)
    parser.add_argument('--generations', type=int, default=3)
    args = parser.parse_args()
    
    evolve_architecture(args)
