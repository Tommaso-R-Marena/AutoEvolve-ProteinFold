"""Multiple Sequence Alignment (MSA) features for improved prediction.

MSA provides evolutionary information that significantly improves accuracy.
This module implements MSA processing similar to AlphaFold2.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np

class MSAProcessor(nn.Module):
    """Process MSA into features for the model.
    
    MSA provides evolutionary covariation information that helps
    predict residue-residue contacts and overall structure.
    """
    
    def __init__(self, msa_dim: int = 64, max_seqs: int = 128):
        super().__init__()
        self.msa_dim = msa_dim
        self.max_seqs = max_seqs
        
        # Embed MSA sequences
        self.msa_embedding = nn.Embedding(21, msa_dim)  # 20 AA + gap
        
        # Row attention (per sequence)
        self.row_attention = nn.MultiheadAttention(msa_dim, 8, batch_first=True)
        
        # Column attention (per position)
        self.col_attention = nn.MultiheadAttention(msa_dim, 8, batch_first=True)
        
        self.norm1 = nn.LayerNorm(msa_dim)
        self.norm2 = nn.LayerNorm(msa_dim)
        
    def forward(self, msa: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process MSA.
        
        Args:
            msa: [B, N_seqs, L] - Multiple sequence alignment
        
        Returns:
            features: Dict with processed MSA features
        """
        batch_size, n_seqs, seq_len = msa.shape
        
        # Embed MSA
        msa_embed = self.msa_embedding(msa)  # [B, N_seqs, L, D]
        
        # Row attention (across residues within each sequence)
        batch_n_seqs = batch_size * n_seqs
        msa_rows = msa_embed.view(batch_n_seqs, seq_len, self.msa_dim)
        msa_rows = self.norm1(msa_rows + self.row_attention(msa_rows, msa_rows, msa_rows)[0])
        msa_embed = msa_rows.view(batch_size, n_seqs, seq_len, self.msa_dim)
        
        # Column attention (across sequences at each position)
        batch_l = batch_size * seq_len
        msa_cols = msa_embed.permute(0, 2, 1, 3).reshape(batch_l, n_seqs, self.msa_dim)
        msa_cols = self.norm2(msa_cols + self.col_attention(msa_cols, msa_cols, msa_cols)[0])
        msa_embed = msa_cols.view(batch_size, seq_len, n_seqs, self.msa_dim).permute(0, 2, 1, 3)
        
        # Extract primary sequence features (first row)
        primary_features = msa_embed[:, 0]  # [B, L, D]
        
        # Compute coevolution features (pair features)
        coevolution = self._compute_coevolution(msa_embed)
        
        return {
            'primary_features': primary_features,
            'coevolution': coevolution,
            'full_msa': msa_embed
        }
    
    def _compute_coevolution(self, msa_embed: torch.Tensor) -> torch.Tensor:
        """Compute pairwise coevolution features.
        
        Uses outer product mean to capture correlations.
        """
        batch_size, n_seqs, seq_len, msa_dim = msa_embed.shape
        
        # Average over sequences
        msa_avg = msa_embed.mean(dim=1)  # [B, L, D]
        
        # Outer product for pairwise features
        left = msa_avg.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [B, L, L, D]
        right = msa_avg.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [B, L, L, D]
        
        # Concatenate and project
        coevolution = torch.cat([left, right, left * right], dim=-1)  # [B, L, L, 3D]
        
        return coevolution

class MSAFeatureExtractor:
    """Extract MSA from sequence databases.
    
    This is a placeholder for actual MSA generation.
    In practice, you'd use tools like HHblits, JackHMMER, or MMseqs2.
    """
    
    @staticmethod
    def generate_dummy_msa(sequence: str, n_seqs: int = 64) -> np.ndarray:
        """Generate dummy MSA for testing.
        
        In production, replace with actual MSA search.
        """
        from model.data_generator import ProteinDataGenerator
        aa_to_idx = ProteinDataGenerator.AMINO_ACID_TO_IDX
        
        seq_len = len(sequence)
        msa = np.zeros((n_seqs, seq_len), dtype=np.int64)
        
        # First row is the query sequence
        msa[0] = [aa_to_idx.get(aa, 0) for aa in sequence]
        
        # Generate homologous sequences with mutations
        for i in range(1, n_seqs):
            msa[i] = msa[0].copy()
            # Randomly mutate 10-30% of positions
            n_mutations = np.random.randint(int(seq_len * 0.1), int(seq_len * 0.3))
            mut_positions = np.random.choice(seq_len, n_mutations, replace=False)
            msa[i][mut_positions] = np.random.randint(0, 20, n_mutations)
        
        return msa
    
    @staticmethod
    def compute_conservation(msa: np.ndarray) -> np.ndarray:
        """Compute per-position conservation scores.
        
        Higher scores = more conserved = likely important.
        """
        n_seqs, seq_len = msa.shape
        conservation = np.zeros(seq_len)
        
        for pos in range(seq_len):
            # Shannon entropy
            residues = msa[:, pos]
            unique, counts = np.unique(residues, return_counts=True)
            probs = counts / n_seqs
            entropy = -np.sum(probs * np.log2(probs + 1e-9))
            conservation[pos] = 1.0 - (entropy / np.log2(20))  # Normalize
        
        return conservation
