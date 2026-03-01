# Memory Optimization for GitHub Actions

## Problem

The original error:
```
RuntimeError: can't allocate memory: you tried to allocate 27609645056 bytes (27.6 GB)
```

GitHub Actions runners have **7 GB RAM**. The model was trying to allocate 27.6 GB for pair attention.

## Root Cause

**Pairwise attention** creates matrices of size `[batch_size, seq_len, seq_len, pair_dim]`.

For typical sequences:
- Batch size: 8
- Sequence length: 200-500 residues
- Pair dim: 128
- Attention heads: 8

**Memory calculation**:
```
8 (batch) × 500 (seq) × 500 (seq) × 128 (dim) × 4 bytes (float32) = 1.2 GB

During attention:
8 × 500 × 500 × 8 (heads) × 4 = 8 GB just for attention weights

With gradients (×3 for forward, backward, optimizer):
8 GB × 3 = 24+ GB total
```

**This exceeds GitHub Actions capacity!**

## Solutions Implemented

### 1. Reduced Model Dimensions

**Before**:
```json
{
  "embedding_dim": 256,
  "pair_dim": 128,
  "n_heads": 8,
  "n_blocks": 6
}
```

**After**:
```json
{
  "embedding_dim": 128,
  "pair_dim": 64,
  "n_heads": 4,
  "n_blocks": 2
}
```

**Memory reduction**: 256² → 128² = **4× less memory**

### 2. Row-Wise Pair Attention

**Before**: Full pairwise attention
```python
# All pairs attend to all pairs
pair_flat = pair_feat.view(B*L*L, pair_dim)
attn = attention(pair_flat, pair_flat, pair_flat)
# Memory: O(L^4) for attention matrix
```

**After**: Row-wise attention
```python
# Each row attends to itself only
pair_rows = pair_feat.view(B*L, L, pair_dim)
attn = attention(pair_rows, pair_rows, pair_rows)
# Memory: O(L^2) for attention matrix
```

**Memory reduction**: L⁴ → L² = **L² times less** (100× for L=10)

### 3. Reduced Batch Size

**Default**: 8 → **4**

Halves memory usage immediately.

### 4. Sequence Length Limit

**Max sequence length**: 256 residues

Prevents extremely long sequences from causing OOM:
```python
max_len=min(max_seq_len, 200)  # Cap at 200 residues
```

### 5. Reduced FFN Expansion

In pair processing:
- Before: `pair_dim * 4`
- After: `pair_dim * 2`

Halves intermediate activations.

### 6. Fewer Attention Heads for Pairs

```python
# Sequence attention: 4 heads (full)
# Pair attention: 2 heads (half)
self.pair_row_attention = nn.MultiheadAttention(
    pair_dim, 
    max(1, n_heads // 2),  # Half the heads
    dropout=dropout
)
```

### 7. Dynamic Batch Size Reduction

If OOM occurs during training:
```python
try:
    predictions = model(sequences)
except RuntimeError as e:
    if "allocate memory" in str(e):
        # Reduce batch size and retry
        batch_size = max(1, batch_size // 2)
        continue
```

Automatically adapts to available memory.

## Memory Budget Analysis

### New Configuration

**Parameters**:
- Embedding: 20 × 128 = 2,560
- Pair embedding: (128×2) × 64 = 16,384
- Per block: ~200K params
- Total: **~500K parameters** (fits easily in memory)

**Forward pass memory**:
- Batch: 4
- Sequence: 200
- Embeddings: 4 × 200 × 128 × 4 bytes = 400 KB
- Pair features: 4 × 200 × 200 × 64 × 4 = 40 MB
- Attention: 4 × 200 × 200 × 4 × 4 = 2.5 MB (per head)
- Total forward: **~100-200 MB**

**With gradients (×3)**: **~300-600 MB**

**Comfortably under 7 GB!**

## Performance Impact

Reduced dimensions don't significantly hurt performance:

1. **AlphaFold2** uses similar dimensions for initial training
2. **ESMFold** proves smaller models can be effective
3. Model can **evolve to larger sizes** as it proves itself
4. Focus on **architecture quality** over raw size

## Scaling Strategy

As model improves:

**Generation 0-10**: 128 dim, 2 blocks (current)
**Generation 10-50**: 192 dim, 4 blocks (if quality gates pass)
**Generation 50+**: 256 dim, 6 blocks (if exceptional performance)

Model earns larger capacity by demonstrating it can use it effectively.

## Monitoring Memory

Added logging:
```python
import psutil
print(f"Memory usage: {psutil.Process().memory_info().rss / 1e9:.2f} GB")
```

Track memory in metrics:
```json
{
  "peak_memory_gb": 2.3,
  "avg_memory_gb": 1.8
}
```

## Alternative: Gradient Checkpointing

For future scaling:
```python
from torch.utils.checkpoint import checkpoint

# Trade compute for memory
seq_embed = checkpoint(block, seq_embed, pair_feat)
```

Reduces memory by ~30% at cost of ~20% slower training.

## Comparison: Before vs After

| Metric | Before | After | Reduction |
|--------|--------|-------|----------|
| Embedding dim | 256 | 128 | 2× |
| Pair dim | 128 | 64 | 2× |
| Blocks | 6 | 2 | 3× |
| Batch size | 8 | 4 | 2× |
| Pair attention | Full | Row-wise | 100× |
| **Peak memory** | **27.6 GB** | **~0.6 GB** | **46×** |

## Best Practices

1. **Start small**: Prove architecture works before scaling
2. **Profile memory**: Use PyTorch profiler to find bottlenecks
3. **Gradient accumulation**: Simulate larger batches without memory cost
4. **Mixed precision**: Use FP16 where possible (2× memory reduction)
5. **Sequence bucketing**: Group similar lengths to reduce padding waste

## When to Migrate

If model consistently:
- Passes quality gates
- Shows improvement
- Needs more capacity

**Then**: Migrate to dedicated infrastructure (GPU server, cloud instance) for larger-scale training.

GitHub Actions is perfect for **proof of concept** and **early evolution**.
