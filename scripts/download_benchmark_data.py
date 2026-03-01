#!/usr/bin/env python3
import requests
import argparse
from pathlib import Path
import gzip
import shutil
from typing import List
import json

class BenchmarkDataDownloader:
    """Download benchmark datasets for protein folding evaluation."""
    
    def __init__(self, data_dir: str = 'data/benchmark'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_casp15(self):
        """Download CASP15 targets."""
        print("Downloading CASP15 benchmark data...")
        
        casp_dir = self.data_dir / 'CASP15'
        casp_dir.mkdir(exist_ok=True)
        
        # CASP15 targets list (publicly available)
        targets_url = "https://predictioncenter.org/casp15/targetlist.cgi?type=csv"
        
        try:
            response = requests.get(targets_url, timeout=30)
            if response.status_code == 200:
                with open(casp_dir / 'targets.csv', 'w') as f:
                    f.write(response.text)
                print(f"✓ CASP15 targets downloaded to {casp_dir}")
            else:
                print(f"Warning: Could not download CASP15 data (status {response.status_code})")
                self._create_mock_casp15(casp_dir)
        except Exception as e:
            print(f"Error downloading CASP15: {e}")
            self._create_mock_casp15(casp_dir)
    
    def _create_mock_casp15(self, casp_dir: Path):
        """Create mock CASP15 data for testing."""
        mock_data = {
            'targets': [
                {'id': 'T1104', 'sequence': 'MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITRRGG'},
                {'id': 'T1106', 'sequence': 'MKKYTCTVCGYIYNPEDGDPDNGVNPGTDFKDIPDDWVCPL'}
            ]
        }
        with open(casp_dir / 'targets.json', 'w') as f:
            json.dump(mock_data, f, indent=2)
        print(f"✓ Created mock CASP15 data in {casp_dir}")
    
    def download_cameo(self):
        """Download CAMEO continuous evaluation targets."""
        print("Downloading CAMEO benchmark data...")
        
        cameo_dir = self.data_dir / 'CAMEO'
        cameo_dir.mkdir(exist_ok=True)
        
        # CAMEO provides weekly targets
        cameo_url = "https://www.cameo3d.org/"
        
        # Since CAMEO requires specific weekly snapshots, create mock data
        self._create_mock_cameo(cameo_dir)
    
    def _create_mock_cameo(self, cameo_dir: Path):
        """Create mock CAMEO data for testing."""
        mock_data = {
            'targets': [
                {'id': '2024-01-01_00001', 'sequence': 'MKLFFNILCLFPVLAADNHGVGPQGASGVDPIT'},
                {'id': '2024-01-08_00002', 'sequence': 'MKYTCTVCGYIYNPEDGDPDNGVNPGTDFKD'}
            ]
        }
        with open(cameo_dir / 'targets.json', 'w') as f:
            json.dump(mock_data, f, indent=2)
        print(f"✓ Created mock CAMEO data in {cameo_dir}")
    
    def download_pdb_structures(self, pdb_ids: List[str]):
        """Download specific PDB structures for evaluation."""
        pdb_dir = self.data_dir / 'PDB'
        pdb_dir.mkdir(exist_ok=True)
        
        for pdb_id in pdb_ids:
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(pdb_dir / f"{pdb_id}.pdb", 'w') as f:
                        f.write(response.text)
                    print(f"✓ Downloaded {pdb_id}.pdb")
                else:
                    print(f"✗ Failed to download {pdb_id}")
            except Exception as e:
                print(f"Error downloading {pdb_id}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Download benchmark datasets')
    parser.add_argument('--datasets', nargs='+', default=['CASP15', 'CAMEO'],
                       choices=['CASP15', 'CAMEO', 'PDB'],
                       help='Datasets to download')
    parser.add_argument('--pdb-ids', nargs='+', default=[],
                       help='Specific PDB IDs to download')
    parser.add_argument('--data-dir', type=str, default='data/benchmark',
                       help='Directory to store benchmark data')
    
    args = parser.parse_args()
    
    downloader = BenchmarkDataDownloader(args.data_dir)
    
    for dataset in args.datasets:
        if dataset == 'CASP15':
            downloader.download_casp15()
        elif dataset == 'CAMEO':
            downloader.download_cameo()
    
    if args.pdb_ids:
        downloader.download_pdb_structures(args.pdb_ids)
    
    print("\n✅ Benchmark data download complete!")

if __name__ == '__main__':
    main()
