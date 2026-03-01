#!/usr/bin/env python3
import torch
import argparse
from pathlib import Path

def verify_weights(args):
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"Checkpoint {checkpoint_path} not found")
        exit(1)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        required_keys = ['model_state_dict', 'config', 'generation']
        for key in required_keys:
            if key not in checkpoint:
                print(f"Missing required key: {key}")
                exit(1)
        
        print(f"Checkpoint verified successfully")
        print(f"Generation: {checkpoint['generation']}")
        print(f"Config keys: {list(checkpoint['config'].keys())}")
        
    except Exception as e:
        print(f"Error verifying checkpoint: {e}")
        exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    verify_weights(args)
