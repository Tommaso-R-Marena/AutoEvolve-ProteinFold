"""Neural Architecture Search (NAS) for automatic architecture optimization.

Implements a differentiable architecture search strategy that learns optimal:
- Number of layers
- Layer types (attention, convolution, etc.)
- Connection patterns
- Hyperparameters
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np

class SearchableOperation(nn.Module):
    """A single operation in the search space."""
    
    def __init__(self, dim: int, operation_type: str):
        super().__init__()
        self.operation_type = operation_type
        
        if operation_type == 'attention':
            self.op = nn.MultiheadAttention(dim, 8, batch_first=True)
        elif operation_type == 'conv':
            self.op = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        elif operation_type == 'gated_ffn':
            self.op = GatedFFN(dim)
        elif operation_type == 'identity':
            self.op = nn.Identity()
        elif operation_type == 'zero':
            self.op = lambda x: torch.zeros_like(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.operation_type == 'attention':
            return self.op(x, x, x)[0]
        elif self.operation_type == 'conv':
            # Conv1d expects [B, C, L]
            x = x.transpose(1, 2)
            x = self.op(x)
            return x.transpose(1, 2)
        else:
            return self.op(x)

class GatedFFN(nn.Module):
    """Gated feed-forward network (SwiGLU-style)."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 4)
        self.w2 = nn.Linear(dim * 4, dim)
        self.w3 = nn.Linear(dim, dim * 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class SearchableBlock(nn.Module):
    """A block with multiple operation choices, learned through architecture parameters."""
    
    OPERATIONS = ['attention', 'conv', 'gated_ffn', 'identity']
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Create all possible operations
        self.operations = nn.ModuleList([
            SearchableOperation(dim, op_type) for op_type in self.OPERATIONS
        ])
        
        # Architecture parameters (learned)
        self.arch_params = nn.Parameter(torch.randn(len(self.OPERATIONS)))
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute operation weights
        weights = F.softmax(self.arch_params, dim=0)
        
        # Weighted combination of all operations
        output = sum(w * op(x) for w, op in zip(weights, self.operations))
        return self.norm(x + output)
    
    def get_best_operation(self) -> str:
        """Return the operation with highest weight."""
        best_idx = self.arch_params.argmax().item()
        return self.OPERATIONS[best_idx]

class NASProteinModel(nn.Module):
    """Protein folding model with neural architecture search.
    
    The architecture is learned jointly with the weights during training.
    After search, the best architecture can be extracted and used.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        dim = config['embedding_dim']
        
        # Embeddings
        self.amino_acid_embedding = nn.Embedding(config['vocab_size'], dim)
        
        # Searchable blocks
        self.num_blocks = config.get('nas_blocks', 6)
        self.blocks = nn.ModuleList([
            SearchableBlock(dim) for _ in range(self.num_blocks)
        ])
        
        # Output heads
        self.coord_head = nn.Linear(dim, 3)
        self.confidence_head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.amino_acid_embedding(sequence)
        
        for block in self.blocks:
            x = block(x)
        
        coords = self.coord_head(x)
        confidence = self.confidence_head(x).squeeze(-1)
        
        return {
            'coordinates': coords,
            'confidence': confidence,
            'embeddings': x
        }
    
    def get_architecture_summary(self) -> Dict[int, str]:
        """Get the learned architecture."""
        return {
            i: block.get_best_operation() 
            for i, block in enumerate(self.blocks)
        }
    
    def architecture_parameters(self) -> List[nn.Parameter]:
        """Get architecture parameters for separate optimization."""
        return [block.arch_params for block in self.blocks]
    
    def model_parameters(self) -> List[nn.Parameter]:
        """Get model parameters (excluding architecture parameters)."""
        arch_params = set(self.architecture_parameters())
        return [p for p in self.parameters() if p not in arch_params]

class NASTrainer:
    """Trainer for neural architecture search."""
    
    def __init__(self, model: NASProteinModel, lr_model: float = 1e-4, lr_arch: float = 3e-4):
        self.model = model
        
        # Separate optimizers for model and architecture
        self.model_optimizer = torch.optim.Adam(model.model_parameters(), lr=lr_model)
        self.arch_optimizer = torch.optim.Adam(model.architecture_parameters(), lr=lr_arch)
    
    def train_step(self, train_batch: Dict, val_batch: Dict, criterion: nn.Module) -> Dict[str, float]:
        """One training step with architecture search.
        
        Alternates between:
        1. Update architecture parameters on validation data
        2. Update model parameters on training data
        """
        losses = {}
        
        # Step 1: Update architecture
        self.arch_optimizer.zero_grad()
        val_predictions = self.model(val_batch['sequences'])
        arch_loss = criterion(val_predictions, val_batch)
        arch_loss.backward()
        self.arch_optimizer.step()
        losses['arch_loss'] = arch_loss.item()
        
        # Step 2: Update model
        self.model_optimizer.zero_grad()
        train_predictions = self.model(train_batch['sequences'])
        model_loss = criterion(train_predictions, train_batch)
        model_loss.backward()
        self.model_optimizer.step()
        losses['model_loss'] = model_loss.item()
        
        return losses
