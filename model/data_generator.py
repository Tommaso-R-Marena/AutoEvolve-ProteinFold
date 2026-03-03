import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import requests
import json
from Bio import SeqIO
from io import StringIO
import random
import time

class ProteinDataGenerator:
    """Generate synthetic and real protein data for training."""
    
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    AMINO_ACID_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
    
    # High-coverage organisms in AlphaFold Database
    HIGH_COVERAGE_ORGANISMS = [
        ('9606', 'Homo sapiens'),           # Human - ~20K proteins
        ('10090', 'Mus musculus'),          # Mouse - ~22K proteins
        ('7227', 'Drosophila melanogaster'), # Fruit fly - ~14K proteins
        ('6239', 'Caenorhabditis elegans'), # C. elegans - ~20K proteins
        ('559292', 'Saccharomyces cerevisiae'), # Yeast - ~6K proteins
        ('83333', 'Escherichia coli'),      # E. coli - ~4K proteins
        ('3702', 'Arabidopsis thaliana'),   # Plant model - ~27K proteins
    ]
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self.api_keys = api_keys or {}
        self.cache = {}
        self.session = requests.Session()  # Reuse connection
        self.session.headers.update({'User-Agent': 'AutoEvolve-ProteinFold/1.0'})
        
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
            'A': 0.0825, 'C': 0.0137, 'D': 0.0545, 'E': 0.0675, 'F': 0.0386,
            'G': 0.0708, 'H': 0.0227, 'I': 0.0596, 'K': 0.0584, 'L': 0.0966,
            'M': 0.0242, 'N': 0.0406, 'P': 0.0470, 'Q': 0.0393, 'R': 0.0553,
            'S': 0.0656, 'T': 0.0534, 'V': 0.0687, 'W': 0.0108, 'Y': 0.0292
        }
        
        amino_acids = list(aa_freq.keys())
        probabilities = [aa_freq[aa] for aa in amino_acids]
        
        # Normalize probabilities to ensure they sum to exactly 1.0
        prob_sum = sum(probabilities)
        probabilities = [p / prob_sum for p in probabilities]
        
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
    
    def fetch_real_data_uniprot(self, n_samples: int = 10, prefer_high_coverage: bool = True) -> List[Dict]:
        """Fetch real protein sequences from UniProt API.
        
        Args:
            n_samples: Number of sequences to fetch
            prefer_high_coverage: If True, prioritize organisms with high AlphaFold coverage
        
        Returns:
            List of protein data dictionaries
        """
        data = []
        
        if prefer_high_coverage:
            # Distribute samples across high-coverage organisms
            samples_per_org = max(1, n_samples // len(self.HIGH_COVERAGE_ORGANISMS))
            
            for organism_id, organism_name in self.HIGH_COVERAGE_ORGANISMS:
                try:
                    org_data = self._fetch_from_organism(organism_id, samples_per_org)
                    data.extend(org_data)
                    
                    if len(data) >= n_samples:
                        break
                except Exception as e:
                    print(f"⚠️  Failed to fetch from {organism_name}: {e}")
                    continue
        else:
            # Fetch random reviewed proteins
            data = self._fetch_random_proteins(n_samples)
        
        return data[:n_samples]
    
    def _fetch_from_organism(self, organism_id: str, n_samples: int) -> List[Dict]:
        """Fetch proteins from specific organism.
        
        CRITICAL: Query reviewed:true to get SwissProt proteins with high AlphaFold coverage.
        """
        data = []
        
        try:
            # Query REVIEWED proteins only (SwissProt, not TrEMBL)
            # Added 'existence:1' to get proteins with evidence at protein level
            url = "https://rest.uniprot.org/uniprotkb/stream"
            params = {
                'format': 'fasta',
                'query': f'organism_id:{organism_id} AND reviewed:true AND existence:1',
                'size': n_samples * 3  # Fetch extra to filter later
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse FASTA
            fasta_io = StringIO(response.text)
            for record in SeqIO.parse(fasta_io, "fasta"):
                # Extract clean UniProt ID (sp|P12345|NAME format)
                parts = record.id.split('|')
                if len(parts) >= 2:
                    uniprot_id = parts[1]  # Middle part is the accession
                else:
                    uniprot_id = record.id
                
                # Skip fragment proteins (accession starts with A0A or has -# suffix)
                if uniprot_id.startswith('A0A') or '-' in uniprot_id:
                    continue
                
                # Skip if too short or too long
                seq_len = len(str(record.seq))
                if seq_len < 30 or seq_len > 1000:
                    continue
                
                data.append({
                    'id': uniprot_id,
                    'full_id': record.id,
                    'sequence': str(record.seq),
                    'description': record.description,
                    'organism_id': organism_id
                })
                
                if len(data) >= n_samples:
                    break
        
        except Exception as e:
            print(f"Error fetching from organism {organism_id}: {e}")
            return []
        
        return data
    
    def _fetch_random_proteins(self, n_samples: int) -> List[Dict]:
        """Fetch random REVIEWED proteins with evidence at protein level."""
        data = []
        
        try:
            url = "https://rest.uniprot.org/uniprotkb/stream"
            params = {
                'format': 'fasta',
                'query': 'reviewed:true AND existence:1',
                'size': n_samples * 3
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse FASTA
            fasta_io = StringIO(response.text)
            for record in SeqIO.parse(fasta_io, "fasta"):
                parts = record.id.split('|')
                uniprot_id = parts[1] if len(parts) >= 2 else record.id
                
                # Skip fragments
                if uniprot_id.startswith('A0A') or '-' in uniprot_id:
                    continue
                
                # Skip if too short or too long
                seq_len = len(str(record.seq))
                if seq_len < 30 or seq_len > 1000:
                    continue
                
                data.append({
                    'id': uniprot_id,
                    'full_id': record.id,
                    'sequence': str(record.seq),
                    'description': record.description
                })
                
                if len(data) >= n_samples:
                    break
        
        except Exception as e:
            print(f"Error fetching random proteins: {e}")
            return []
        
        return data
    
    def fetch_alphafold_structure(self, uniprot_id: str, retry: int = 2) -> Optional[np.ndarray]:
        """Fetch predicted structure from AlphaFold DB with version fallback.
        
        AlphaFold DB updated to v6 in October 2025. This function tries:
        1. v6 (current, 241M predictions)
        2. v4 (legacy, 214M predictions) 
        3. v2 (original release)
        
        Args:
            uniprot_id: UniProt accession ID
            retry: Number of retry attempts per version
        
        Returns:
            Numpy array of C-alpha coordinates or None
        """
        # Try versions in order: v6 (latest) → v4 → v2
        versions = ['v6', 'v5', 'v4', 'v3', 'v2', 'v1']
        
        for version in versions:
            for attempt in range(retry):
                try:
                    # AlphaFold DB file URL format
                    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_{version}.pdb"
                    response = self.session.get(url, timeout=15)
                    
                    if response.status_code == 200:
                        # Parse PDB and extract C-alpha coordinates
                        coords = self._parse_pdb_coords(response.text)
                        if len(coords) > 0:
                            return coords
                    elif response.status_code == 404:
                        # Not found in this version, try next
                        break
                    elif response.status_code == 429:
                        # Rate limited, wait longer
                        time.sleep(2.0)
                        continue
                    
                    # Other error, retry with backoff
                    if attempt < retry - 1:
                        time.sleep(0.5 * (attempt + 1))
                
                except requests.exceptions.Timeout:
                    if attempt < retry - 1:
                        time.sleep(1.0)
                    continue
                except requests.exceptions.RequestException as e:
                    # Connection error, retry
                    if attempt < retry - 1:
                        time.sleep(0.5)
                    continue
                except Exception as e:
                    # Unexpected error, skip this version
                    break
        
        return None  # Not found in any version
    
    def _parse_pdb_coords(self, pdb_text: str) -> np.ndarray:
        """Extract C-alpha coordinates from PDB text."""
        coords = []
        for line in pdb_text.split('\n'):
            if line.startswith('ATOM') and ' CA ' in line:
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coords.append([x, y, z])
                except (ValueError, IndexError):
                    continue
        return np.array(coords) if coords else np.array([])
    
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
    
    def get_organism_stats(self) -> Dict:
        """Get statistics about high-coverage organisms."""
        return {
            'high_coverage_organisms': [
                {'id': org_id, 'name': name} 
                for org_id, name in self.HIGH_COVERAGE_ORGANISMS
            ],
            'total_organisms': len(self.HIGH_COVERAGE_ORGANISMS)
        }
