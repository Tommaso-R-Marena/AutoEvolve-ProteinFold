import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import requests
import json
from Bio import SeqIO
from io import StringIO
import random

class ProteinDataGenerator:
    """Generate synthetic and real protein data for training."""
    
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    AMINO_ACID_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self.api_keys = api_keys or {}
        self.cache = {}
        
    def generate_synthetic_batch(self, batch_size: int, min_len: int = 50, max_len: int = 500) -> Dict[str, torch.Tensor]:
        """Generate physically plausible synthetic protein sequences."""
        sequences = []
        lengths = []
        
        for _ in range(batch_size):
            length = random.randint(min_len, max_len)
            lengths.append(length)
            
            # Generate sequence with amino acid frequency bias
            sequence = self._generate_realistic_sequence(length)
            sequences.append(sequence)
        
        # Pad sequences
        max_len_batch = max(lengths)
        padded_seqs = torch.zeros(batch_size, max_len_batch, dtype=torch.long)
        
        for i, seq in enumerate(sequences):
            padded_seqs[i, :lengths[i]] = torch.tensor(
                [self.AMINO_ACID_TO_IDX[aa] for aa in seq],
                dtype=torch.long
            )
        
        # Generate synthetic structures (random walk with constraints)
        coords = self._generate_synthetic_structure(padded_seqs, lengths)
        
        return {
            'sequences': padded_seqs,
            'coordinates': coords,
            'lengths': torch.tensor(lengths),
            'mask': self._create_mask(lengths, max_len_batch)
        }
    
    def _generate_realistic_sequence(self, length: int) -> str:
        """Generate sequence with realistic amino acid frequencies."""
        # Approximate amino acid frequencies in proteins
        aa_freq = {
            'A': 0.08, 'C': 0.02, 'D': 0.05, 'E': 0.06, 'F': 0.04,
            'G': 0.07, 'H': 0.02, 'I': 0.05, 'K': 0.06, 'L': 0.10,
            'M': 0.02, 'N': 0.04, 'P': 0.05, 'Q': 0.04, 'R': 0.05,
            'S': 0.07, 'T': 0.06, 'V': 0.07, 'W': 0.01, 'Y': 0.03
        }
        
        amino_acids = list(aa_freq.keys())
        probabilities = [aa_freq[aa] for aa in amino_acids]
        
        sequence = ''.join(np.random.choice(amino_acids, size=length, p=probabilities))
        return sequence
    
    def _generate_synthetic_structure(self, sequences: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        """Generate synthetic 3D coordinates with protein-like geometry."""
        batch_size, max_len = sequences.shape
        coords = torch.zeros(batch_size, max_len, 3)
        
        for i, length in enumerate(lengths):
            # Generate backbone with realistic bond lengths and angles
            for j in range(length):
                if j == 0:
                    coords[i, j] = torch.randn(3) * 0.5
                else:
                    # Constrained random walk (approximate C-alpha distance ~3.8Å)
                    direction = torch.randn(3)
                    direction = direction / torch.norm(direction)
                    coords[i, j] = coords[i, j-1] + direction * (3.8 + torch.randn(1) * 0.2)
        
        return coords
    
    def fetch_real_data_uniprot(self, n_samples: int = 10) -> List[Dict]:
        """Fetch real protein sequences from UniProt API."""
        data = []
        
        try:
            # UniProt REST API query for reviewed proteins
            url = f"https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=reviewed:true&size={n_samples}"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse FASTA
            fasta_io = StringIO(response.text)
            for record in SeqIO.parse(fasta_io, "fasta"):
                data.append({
                    'id': record.id,
                    'sequence': str(record.seq),
                    'description': record.description
                })
                
                if len(data) >= n_samples:
                    break
        
        except Exception as e:
            print(f"Error fetching UniProt data: {e}")
            # Return synthetic data as fallback
            return []
        
        return data
    
    def fetch_alphafold_structure(self, uniprot_id: str) -> Optional[np.ndarray]:
        """Fetch predicted structure from AlphaFold DB."""
        try:
            url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # Parse PDB and extract C-alpha coordinates
                coords = self._parse_pdb_coords(response.text)
                return coords
        
        except Exception as e:
            print(f"Error fetching AlphaFold structure for {uniprot_id}: {e}")
        
        return None
    
    def _parse_pdb_coords(self, pdb_text: str) -> np.ndarray:
        """Extract C-alpha coordinates from PDB text."""
        coords = []
        for line in pdb_text.split('\n'):
            if line.startswith('ATOM') and ' CA ' in line:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                coords.append([x, y, z])
        return np.array(coords)
    
    def _create_mask(self, lengths: List[int], max_len: int) -> torch.Tensor:
        """Create attention mask for padded sequences."""
        batch_size = len(lengths)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, :length] = True
        return mask
    
    def augment_data(self, sequences: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation (rotation, noise, cropping)."""
        batch_size = sequences.shape[0]
        
        # Random rotation
        for i in range(batch_size):
            angle = torch.rand(1) * 2 * np.pi
            rotation_matrix = self._get_rotation_matrix(angle)
            coords[i] = torch.matmul(coords[i], rotation_matrix)
        
        # Add small noise
        coords = coords + torch.randn_like(coords) * 0.1
        
        return sequences, coords
    
    def _get_rotation_matrix(self, angle: float) -> torch.Tensor:
        """Generate 3D rotation matrix around random axis."""
        axis = torch.randn(3)
        axis = axis / torch.norm(axis)
        
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Rodrigues' rotation formula
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = torch.eye(3) + sin_angle * K + (1 - cos_angle) * torch.matmul(K, K)
        return R
