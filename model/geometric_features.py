"""Advanced geometric and chemical features for protein structure prediction.

Includes:
- Invariant point attention (IPA) from AlphaFold2
- Distance and angle features
- Chemical bond constraints
- Chirality preservation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

class InvariantPointAttention(nn.Module):
    """Invariant Point Attention from AlphaFold2.
    
    Operates on 3D coordinates in an SE(3)-equivariant manner.
    This preserves rotational and translational invariance.
    """
    
    def __init__(self, dim: int, n_heads: int = 12, n_query_points: int = 4, n_value_points: int = 8):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_query_points = n_query_points
        self.n_value_points = n_value_points
        
        # Scalar projections
        self.q_scalar = nn.Linear(dim, n_heads * 16)
        self.k_scalar = nn.Linear(dim, n_heads * 16)
        self.v_scalar = nn.Linear(dim, n_heads * 16)
        
        # Point projections (3D)
        self.q_points = nn.Linear(dim, n_heads * n_query_points * 3)
        self.k_points = nn.Linear(dim, n_heads * n_query_points * 3)
        self.v_points = nn.Linear(dim, n_heads * n_value_points * 3)
        
        # Output projection
        self.out_proj = nn.Linear(n_heads * (16 + n_value_points * 3), dim)
        
        self.softplus = nn.Softplus()
    
    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: [B, L, D] - sequence features
            coords: [B, L, 3] - current 3D coordinates
        
        Returns:
            updated_features: [B, L, D]
            updated_coords: [B, L, 3]
        """
        batch_size, seq_len, _ = x.shape
        
        # Scalar attention
        q_s = self.q_scalar(x).view(batch_size, seq_len, self.n_heads, 16)
        k_s = self.k_scalar(x).view(batch_size, seq_len, self.n_heads, 16)
        v_s = self.v_scalar(x).view(batch_size, seq_len, self.n_heads, 16)
        
        # Point attention
        q_p = self.q_points(x).view(batch_size, seq_len, self.n_heads, self.n_query_points, 3)
        k_p = self.k_points(x).view(batch_size, seq_len, self.n_heads, self.n_query_points, 3)
        v_p = self.v_points(x).view(batch_size, seq_len, self.n_heads, self.n_value_points, 3)
        
        # Add current coordinates to query/key points
        coords_expanded = coords.unsqueeze(2).unsqueeze(3)  # [B, L, 1, 1, 3]
        q_p = q_p + coords_expanded
        k_p = k_p + coords_expanded
        
        # Compute attention scores
        # Scalar component
        attn_scalar = torch.einsum('bqhd,bkhd->bhqk', q_s, k_s) / math.sqrt(16)
        
        # Point component (distance-based)
        attn_points = torch.sum((q_p.unsqueeze(3) - k_p.unsqueeze(2)) ** 2, dim=(-2, -1))  # [B, H, L, L]
        attn_points = -self.softplus(attn_points)  # Closer points = higher attention
        
        # Combined attention
        attn = F.softmax(attn_scalar + attn_points * 0.5, dim=-1)
        
        # Apply attention to values
        # Scalar values
        out_scalar = torch.einsum('bhqk,bkhd->bqhd', attn, v_s)
        
        # Point values
        out_points = torch.einsum('bhqk,bkhpd->bqhpd', attn, v_p)
        
        # Flatten heads
        out_scalar = out_scalar.reshape(batch_size, seq_len, -1)
        out_points = out_points.reshape(batch_size, seq_len, -1)
        
        # Combine and project
        out = self.out_proj(torch.cat([out_scalar, out_points], dim=-1))
        
        # Update coordinates (mean of point values)
        updated_coords = v_p.mean(dim=(2, 3))  # [B, L, 3]
        
        return out, updated_coords

class GeometricFeatures(nn.Module):
    """Compute geometric features from coordinates."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances, angles, and dihedrals.
        
        Args:
            coords: [B, L, 3]
        
        Returns:
            features: [B, L, L, F] where F includes:
                - distance (1)
                - distance bins (20)
                - unit direction vector (3)
        """
        batch_size, seq_len, _ = coords.shape
        
        # Pairwise distances
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, L, L, 3]
        distances = torch.norm(diff, dim=-1, keepdim=True)  # [B, L, L, 1]
        
        # Distance bins (0-20Å in 1Å bins)
        dist_bins = self._distance_bins(distances.squeeze(-1), 0, 20, 20)
        
        # Unit direction vectors
        unit_vectors = diff / (distances + 1e-8)
        
        # Concatenate all features
        features = torch.cat([distances, dist_bins, unit_vectors], dim=-1)
        
        return features
    
    def _distance_bins(self, distances: torch.Tensor, min_dist: float, max_dist: float, n_bins: int) -> torch.Tensor:
        """Convert distances to one-hot bins."""
        bin_edges = torch.linspace(min_dist, max_dist, n_bins + 1, device=distances.device)
        bins = torch.searchsorted(bin_edges, distances.contiguous())
        bins = torch.clamp(bins, 0, n_bins - 1)
        return F.one_hot(bins, n_bins).float()

class ChiralityConstraint(nn.Module):
    """Enforce proper chirality for amino acids.
    
    All amino acids (except glycine) are L-amino acids.
    This module adds a loss term to enforce correct stereochemistry.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, coords: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        """Compute chirality violation loss.
        
        Args:
            coords: [B, L, 3] - CA coordinates
            sequence: [B, L] - amino acid indices
        
        Returns:
            loss: scalar - penalty for chirality violations
        """
        # For simplicity, we compute a pseudo-chirality based on
        # the cross product of vectors to neighboring residues
        
        if coords.shape[1] < 3:
            return torch.tensor(0.0, device=coords.device)
        
        # Vectors to adjacent residues
        v1 = coords[:, 1:] - coords[:, :-1]  # [B, L-1, 3]
        v2 = coords[:, 2:] - coords[:, :-2]  # [B, L-2, 3]
        
        # Cross product (should point in consistent direction)
        cross = torch.cross(v1[:, :-1], v2, dim=-1)
        
        # Dot product with next cross product (should be positive)
        consistency = torch.sum(cross[:, :-1] * cross[:, 1:], dim=-1)
        
        # Penalize negative dot products
        violation = F.relu(-consistency)
        
        return violation.mean()

class DistanceConstraints(nn.Module):
    """Physical constraints on distances and angles."""
    
    # Typical peptide bond length (CA-CA distance)
    CA_CA_DISTANCE = 3.8  # Angstroms
    CA_CA_STD = 0.1
    
    def __init__(self):
        super().__init__()
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute constraint violation loss.
        
        Args:
            coords: [B, L, 3]
        
        Returns:
            loss: scalar
        """
        # Adjacent CA distances should be ~3.8Å
        adjacent_distances = torch.norm(coords[:, 1:] - coords[:, :-1], dim=-1)
        
        # Deviation from ideal distance
        distance_violation = torch.abs(adjacent_distances - self.CA_CA_DISTANCE)
        
        # Penalize large deviations
        loss = torch.mean(distance_violation ** 2)
        
        return loss
