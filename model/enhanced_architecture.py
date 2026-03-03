"""Enhanced architecture combining all advanced techniques.

Integrates:
- Diffusion-based generation
- Invariant point attention
- Geometric features
- Physical constraints
- Original Evoformer blocks
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from model.architecture import EvolvableProteinFoldingModel, EvoformerBlock
from model.diffusion_module import DiffusionStructurePredictor
from model.geometric_features import InvariantPointAttention, GeometricFeatures, ChiralityConstraint, DistanceConstraints

class RevolutionaryProteinFolder(nn.Module):
    """State-of-the-art protein folding architecture.
    
    Combines multiple cutting-edge techniques:
    1. Evoformer blocks for sequence-structure coupling
    2. Invariant Point Attention for SE(3)-equivariance 
    3. Diffusion-based iterative refinement
    4. Geometric features and physical constraints
    5. Multi-scale processing
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.generation = 0
        self.performance_history = []
        
        dim = config['embedding_dim']
        pair_dim = config['pair_dim']
        
        # Embeddings
        self.amino_acid_embedding = nn.Embedding(config['vocab_size'], dim)
        
        # Position encoding (Rotary)
        self.rotary_emb = RotaryEmbedding(dim)
        
        # Pairwise feature extraction
        pair_input_dim = dim * 2
        self.pair_embedding = nn.Linear(pair_input_dim, pair_dim)
        
        # Geometric features
        self.geometric_features = GeometricFeatures()
        self.geom_proj = nn.Linear(24, pair_dim)  # Project geometric features to pair_dim
        
        # Evoformer blocks
        self.evoformer_blocks = nn.ModuleList([
            EvoformerBlock(dim, pair_dim, config['n_heads'], config['dropout'])
            for _ in range(config['n_blocks'])
        ])
        
        # Invariant Point Attention blocks
        self.ipa_blocks = nn.ModuleList([
            InvariantPointAttention(dim, n_heads=8)
            for _ in range(3)
        ])
        
        # Initial structure prediction (coarse)
        self.initial_structure = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(512, 3)
        )
        
        # Diffusion refinement
        self.use_diffusion = config.get('use_diffusion', True)
        if self.use_diffusion:
            self.diffusion = DiffusionStructurePredictor(dim, pair_dim, timesteps=20)
        
        # Additional outputs
        self.confidence_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.angle_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # phi, psi, omega
        )
        
        # Physical constraints
        self.chirality_constraint = ChiralityConstraint()
        self.distance_constraint = DistanceConstraints()
    
    def forward(self, sequence: torch.Tensor, num_recycles: int = 3) -> Dict[str, torch.Tensor]:
        """Forward pass with recycling.
        
        Args:
            sequence: [B, L] - amino acid indices
            num_recycles: number of times to recycle predictions
        
        Returns:
            Dictionary containing:
                - coordinates: [B, L, 3]
                - confidence: [B, L]
                - angles: [B, L, 3]
                - constraint_losses: dict
        """
        batch_size, seq_len = sequence.shape
        device = sequence.device
        
        # Amino acid embeddings
        seq_embed = self.amino_acid_embedding(sequence)  # [B, L, D]
        seq_embed = self.rotary_emb(seq_embed)
        
        # Initialize pairwise features
        pair_feat = self._compute_pair_features(seq_embed)
        
        # Initialize coordinates (random)
        coords = torch.randn(batch_size, seq_len, 3, device=device)
        
        # Recycling loop
        for recycle in range(num_recycles):
            # Add geometric features from current coordinates
            if recycle > 0:
                geom_feat = self.geometric_features(coords)
                geom_proj = self.geom_proj(geom_feat)
                pair_feat = pair_feat + geom_proj * 0.1
            
            # Evoformer blocks
            for block in self.evoformer_blocks:
                seq_embed, pair_feat = block(seq_embed, pair_feat)
            
            # Invariant Point Attention (structure refinement)
            for ipa_block in self.ipa_blocks:
                delta_embed, delta_coords = ipa_block(seq_embed, coords)
                seq_embed = seq_embed + delta_embed
                coords = coords + delta_coords * 0.1  # Small updates
            
            # Predict structure
            if recycle == 0:
                coords = self.initial_structure(seq_embed)
            else:
                # Refine existing structure
                coords = coords + self.initial_structure(seq_embed) * 0.3
        
        # Final refinement with diffusion (optional)
        if self.use_diffusion and not self.training:
            coords = self.diffusion.sample(seq_embed, pair_feat)
        
        # Predict confidence and angles
        confidence = self.confidence_head(seq_embed).squeeze(-1)
        angles = self.angle_head(seq_embed)
        
        # Compute constraint losses
        constraint_losses = {
            'chirality': self.chirality_constraint(coords, sequence),
            'distance': self.distance_constraint(coords)
        }
        
        return {
            'coordinates': coords,
            'confidence': confidence,
            'angles': angles,
            'embeddings': seq_embed,
            'pair_features': pair_feat,
            'constraint_losses': constraint_losses
        }
    
    def _compute_pair_features(self, seq_embed: torch.Tensor) -> torch.Tensor:
        """Compute pairwise features from sequence embeddings."""
        batch_size, seq_len, embed_dim = seq_embed.shape
        
        # Outer concatenation
        left = seq_embed.unsqueeze(2).expand(-1, -1, seq_len, -1)
        right = seq_embed.unsqueeze(1).expand(-1, seq_len, -1, -1)
        pair_concat = torch.cat([left, right], dim=-1)
        
        # Project to pair_dim
        return self.pair_embedding(pair_concat)
    
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
    def load_checkpoint(cls, path: str) -> 'RevolutionaryProteinFolder':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.generation = checkpoint.get('generation', 0)
        model.performance_history = checkpoint.get('performance_history', [])
        return model

class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE)."""
    
    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding.
        
        Args:
            x: [B, L, D]
        
        Returns:
            x with rotary position encoding: [B, L, D]
        """
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        cos = emb.cos()[None, :, :]
        sin = emb.sin()[None, :, :]
        
        # Apply rotation
        x_rot = x * cos + self._rotate_half(x) * sin
        return x_rot
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([-x2, x1], dim=-1)
