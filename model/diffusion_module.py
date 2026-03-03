"""Diffusion-based structure generation for protein folding.

Implements a denoising diffusion model similar to RFdiffusion and AlphaFold3's approach.
This allows the model to iteratively refine protein structures through a learned denoising process.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple

class DiffusionSchedule:
    """Variance schedule for diffusion process."""
    
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.timesteps = timesteps
        
        # Linear schedule (can be improved to cosine)
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute values for diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Posterior variance
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def add_noise(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x)
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        noisy_x = sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise
        return noisy_x, noise

class DiffusionStructurePredictor(nn.Module):
    """Diffusion model for iterative structure refinement.
    
    This module learns to denoise protein structures, allowing for:
    1. High-quality structure generation
    2. Uncertainty quantification through multiple samples
    3. Conditional generation given sequence constraints
    """
    
    def __init__(self, embedding_dim: int, pair_dim: int, timesteps: int = 50):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.timesteps = timesteps
        self.schedule = DiffusionSchedule(timesteps)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Denoising network
        self.denoiser = nn.ModuleList([
            DenoisingBlock(embedding_dim, pair_dim) for _ in range(4)
        ])
        
        # Final coordinate prediction
        self.coord_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 3)
        )
    
    def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embedding."""
        half_dim = 64
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return self.time_mlp(embeddings)
    
    def forward(self, noisy_coords: torch.Tensor, t: torch.Tensor, 
                seq_embed: torch.Tensor, pair_feat: torch.Tensor) -> torch.Tensor:
        """Predict noise to remove from noisy coordinates."""
        batch_size, seq_len, _ = noisy_coords.shape
        
        # Time conditioning
        time_embed = self.get_time_embedding(t)  # [B, D]
        time_embed = time_embed.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, D]
        
        # Combine noisy coordinates with sequence features
        # Project coordinates to embedding space
        coord_embed = noisy_coords @ torch.randn(3, self.embedding_dim, device=noisy_coords.device) * 0.1
        
        # Condition on sequence and time
        h = seq_embed + coord_embed + time_embed
        
        # Denoising blocks
        for block in self.denoiser:
            h = block(h, pair_feat)
        
        # Predict noise
        predicted_noise = self.coord_head(h)
        return predicted_noise
    
    @torch.no_grad()
    def sample(self, seq_embed: torch.Tensor, pair_feat: torch.Tensor, 
               guidance_scale: float = 1.0) -> torch.Tensor:
        """Generate structure through iterative denoising."""
        batch_size, seq_len, embed_dim = seq_embed.shape
        device = seq_embed.device
        
        # Start from pure noise
        coords = torch.randn(batch_size, seq_len, 3, device=device)
        
        # Iterative denoising
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.forward(coords, t_batch, seq_embed, pair_feat)
            
            # Compute denoised coordinates
            alpha = self.schedule.alphas[t]
            alpha_cumprod = self.schedule.alphas_cumprod[t]
            
            # Denoise step
            coords = (coords - (1 - alpha) / torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha)
            
            # Add noise for next step (except last)
            if t > 0:
                noise = torch.randn_like(coords)
                coords = coords + torch.sqrt(self.schedule.posterior_variance[t]) * noise
        
        return coords

class DenoisingBlock(nn.Module):
    """Denoising transformer block with cross-attention to pair features."""
    
    def __init__(self, embedding_dim: int, pair_dim: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embedding_dim, 8, batch_first=True)
        self.pair_attn = nn.MultiheadAttention(embedding_dim, 8, batch_first=True)
        
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        
        # Project pair features to embedding space
        self.pair_proj = nn.Linear(pair_dim, embedding_dim)
    
    def forward(self, x: torch.Tensor, pair_feat: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x = self.norm1(x + self.self_attn(x, x, x)[0])
        
        # Cross-attention to pair features
        batch_size, seq_len, _ = x.shape
        pair_context = pair_feat.mean(dim=2)  # [B, L, pair_dim]
        pair_context = self.pair_proj(pair_context)  # [B, L, embedding_dim]
        x = self.norm2(x + self.pair_attn(x, pair_context, pair_context)[0])
        
        # FFN
        x = self.norm3(x + self.ffn(x))
        
        return x
