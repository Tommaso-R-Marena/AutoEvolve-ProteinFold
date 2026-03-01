import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import json
import numpy as np

class EvolvableProteinFoldingModel(nn.Module):
    """Self-modifying protein folding architecture with dynamic layer generation."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.generation = 0
        self.performance_history = []
        
        # Core embedding layers
        self.amino_acid_embedding = nn.Embedding(
            config['vocab_size'], 
            config['embedding_dim']
        )
        
        # Pairwise feature extraction
        # Input: concat(left_embed, right_embed) = 2*embedding_dim
        pair_input_dim = config['embedding_dim'] * 2
        self.pair_embedding = nn.Linear(
            pair_input_dim,
            config['pair_dim']
        )
        
        # Evoformer-inspired blocks (dynamically sized)
        self.evoformer_blocks = nn.ModuleList([
            EvoformerBlock(
                config['embedding_dim'],
                config['pair_dim'],
                config['n_heads'],
                config['dropout']
            ) for _ in range(config['n_blocks'])
        ])
        
        # Structure module
        self.structure_module = StructureModule(
            config['embedding_dim'],
            config['pair_dim'],
            config['n_structure_blocks']
        )
        
        # Confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(config['embedding_dim'], 256),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, sequence: torch.Tensor, features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with optional auxiliary features."""
        batch_size, seq_len = sequence.shape
        
        # Amino acid embeddings
        seq_embed = self.amino_acid_embedding(sequence)  # [B, L, D]
        
        # Pairwise representations
        pair_feat = self._compute_pair_features(seq_embed, features)
        
        # Evoformer blocks
        for block in self.evoformer_blocks:
            seq_embed, pair_feat = block(seq_embed, pair_feat)
        
        # Structure prediction
        coords, angles = self.structure_module(seq_embed, pair_feat)
        
        # Confidence scores
        confidence = self.confidence_head(seq_embed).squeeze(-1)
        
        return {
            'coordinates': coords,
            'angles': angles,
            'confidence': confidence,
            'embeddings': seq_embed,
            'pair_features': pair_feat
        }
    
    def _compute_pair_features(self, seq_embed: torch.Tensor, features: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute pairwise features from sequence embeddings."""
        batch_size, seq_len, embed_dim = seq_embed.shape
        
        # Outer concatenation: [B, L, L, 2*D]
        left = seq_embed.unsqueeze(2).expand(-1, -1, seq_len, -1)
        right = seq_embed.unsqueeze(1).expand(-1, seq_len, -1, -1)
        pair_concat = torch.cat([left, right], dim=-1)
        
        # Project to pair_dim
        return self.pair_embedding(pair_concat)
    
    def mutate_architecture(self, mutation_rate: float = 0.1) -> Dict:
        """Evolve architecture by adding/removing layers or modifying hyperparameters."""
        mutations = []
        
        # Potentially add a new Evoformer block
        if torch.rand(1).item() < mutation_rate:
            new_block = EvoformerBlock(
                self.config['embedding_dim'],
                self.config['pair_dim'],
                self.config['n_heads'],
                self.config['dropout']
            )
            self.evoformer_blocks.append(new_block)
            self.config['n_blocks'] += 1
            mutations.append(f"Added Evoformer block (total: {self.config['n_blocks']})")
        
        # Potentially remove underperforming block
        if len(self.evoformer_blocks) > 3 and torch.rand(1).item() < mutation_rate * 0.5:
            self.evoformer_blocks = self.evoformer_blocks[:-1]
            self.config['n_blocks'] -= 1
            mutations.append(f"Removed Evoformer block (total: {self.config['n_blocks']})")
        
        self.generation += 1
        return {'mutations': mutations, 'generation': self.generation}
    
    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None):
        """Save model checkpoint with metadata."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'generation': self.generation,
            'performance_history': self.performance_history,
            'metadata': metadata or {}
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str) -> 'EvolvableProteinFoldingModel':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.generation = checkpoint.get('generation', 0)
        model.performance_history = checkpoint.get('performance_history', [])
        return model


class EvoformerBlock(nn.Module):
    """Evoformer-style block with row/column attention."""
    
    def __init__(self, seq_dim: int, pair_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.row_attention = nn.MultiheadAttention(seq_dim, n_heads, dropout=dropout, batch_first=True)
        self.col_attention = nn.MultiheadAttention(seq_dim, n_heads, dropout=dropout, batch_first=True)
        self.pair_attention = nn.MultiheadAttention(pair_dim, n_heads, dropout=dropout, batch_first=True)
        
        self.seq_norm1 = nn.LayerNorm(seq_dim)
        self.seq_norm2 = nn.LayerNorm(seq_dim)
        self.pair_norm = nn.LayerNorm(pair_dim)
        
        self.seq_ffn = nn.Sequential(
            nn.Linear(seq_dim, seq_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_dim * 4, seq_dim)
        )
        
        self.pair_ffn = nn.Sequential(
            nn.Linear(pair_dim, pair_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim * 4, pair_dim)
        )
        
    def forward(self, seq_feat: torch.Tensor, pair_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sequence attention
        seq_feat = self.seq_norm1(seq_feat + self.row_attention(seq_feat, seq_feat, seq_feat)[0])
        seq_feat = self.seq_norm2(seq_feat + self.seq_ffn(seq_feat))
        
        # Pair attention (flatten spatial dimensions)
        batch_size, seq_len, _, pair_dim = pair_feat.shape
        pair_flat = pair_feat.view(batch_size * seq_len, seq_len, pair_dim)
        pair_flat = self.pair_norm(pair_flat + self.pair_attention(pair_flat, pair_flat, pair_flat)[0])
        pair_feat = pair_flat.view(batch_size, seq_len, seq_len, pair_dim)
        pair_feat = pair_feat + self.pair_ffn(pair_feat)
        
        return seq_feat, pair_feat


class StructureModule(nn.Module):
    """Converts embeddings to 3D coordinates and backbone angles."""
    
    def __init__(self, seq_dim: int, pair_dim: int, n_blocks: int):
        super().__init__()
        self.coord_predictor = nn.Sequential(
            nn.Linear(seq_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3)  # x, y, z coordinates
        )
        
        self.angle_predictor = nn.Sequential(
            nn.Linear(seq_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # phi, psi, omega angles
        )
        
    def forward(self, seq_feat: torch.Tensor, pair_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = self.coord_predictor(seq_feat)
        angles = self.angle_predictor(seq_feat)
        return coords, angles
