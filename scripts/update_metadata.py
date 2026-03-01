#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from datetime import datetime

def update_metadata(args):
    metadata_path = Path('metadata.json')
    
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {
            'created': datetime.now().isoformat(),
            'training_sessions': []
        }
    
    # Add new training session
    session = {
        'timestamp': datetime.now().isoformat(),
        'source': args.source
    }
    
    metadata['training_sessions'].append(session)
    metadata['last_updated'] = datetime.now().isoformat()
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata updated for {args.source} training session")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    args = parser.parse_args()
    
    update_metadata(args)
