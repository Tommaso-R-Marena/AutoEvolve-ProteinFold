# Training State Resumption

## Overview

The model saves complete training state every 100 epochs, allowing seamless resumption from exactly where it left off.

## What is Saved

### Complete Training State

Location: `data/training_state/training_state.pkl`

Contains:
- **Epoch number**: Current training epoch
- **Optimizer state**: All optimizer parameters (learning rate, momentum, etc.)
- **Scheduler state**: Learning rate schedule position
- **Total samples processed**: Running count across all cycles
- **Best loss achieved**: For tracking improvements
- **RNG state**: PyTorch and NumPy random number generators
- **Timestamp**: When state was saved

## Why This Matters

### Problem Without State Resumption

```
Cycle 1: Trains for 5 hours → reaches epoch 1000 → saves checkpoint
Cycle 2: Starts from epoch 0 again → different random seed → inconsistent training
```

### With State Resumption

```
Cycle 1: Trains for 5 hours → reaches epoch 1000 → saves state
Cycle 2: Resumes from epoch 1000 → same optimizer state → continues smoothly
Cycle 3: Resumes from epoch 2000 → accumulated knowledge preserved
```

## Usage

### Automatic (GitHub Actions)

The workflow automatically uses `--resume` flag:

```yaml
python scripts/train_cycle.py --mode continuous --max-time 18000 --resume
```

### Manual

```bash
# First run
python scripts/train_cycle.py --max-time 3600

# Resume from saved state
python scripts/train_cycle.py --max-time 3600 --resume

# Start fresh (ignore saved state)
python scripts/train_cycle.py --max-time 3600  # without --resume
```

## Save Frequency

**Every 100 epochs:**
- Quick save (~0.1 seconds)
- Doesn't interrupt training
- Ensures max 100 epochs of work lost if interrupted

**At end of cycle:**
- Final state save
- Committed to Git
- Available for next cycle

## State Restoration

When resuming, the system restores:

1. **Model weights** from checkpoint
2. **Optimizer state** (momentum buffers, etc.)
3. **Learning rate schedule** position
4. **Epoch counter** for logging
5. **Sample counter** for parameter budget
6. **RNG state** for reproducibility

### Example Output

```
Loading existing model...
Loaded model at generation 3

✓ Resuming from epoch 2847
  Total samples processed: 227,760
  Best loss so far: 8.3421

Starting training for 18000 seconds...
Will save state every 100 epochs for seamless resumption

Epoch 2848 | Loss: 8.2156 | ...
Epoch 2849 | Loss: 8.1973 | ...
```

## Reproducibility

### RNG State Preservation

Both PyTorch and NumPy random states are saved:

```python
rng_state = {
    'torch': torch.get_rng_state(),
    'numpy': np.random.get_state()
}
```

This ensures:
- Same data augmentation sequence
- Same dropout masks
- Same synthetic data generation
- Consistent batch sampling

## Benefits

### Continuous Learning

- **No training interruption**: Each cycle builds on previous
- **Accumulated knowledge**: Sample count grows indefinitely
- **Consistent optimization**: Momentum/Adam states preserved
- **Smooth loss curves**: No sudden jumps between cycles

### Efficiency

- **No wasted computation**: Never re-trains same epochs
- **Optimal learning rate**: Scheduler continues from correct position
- **Better convergence**: Maintains optimization trajectory

### Robustness

- **Crash recovery**: Can resume if workflow fails
- **Hardware migration**: Move training between machines
- **Experimentation**: Save state, try changes, revert if needed

## State Management

### Clearing State

To start completely fresh:

```bash
rm data/training_state/training_state.pkl
```

Or programmatically:

```python
from scripts.train_cycle import TrainingState

state_manager = TrainingState()
state_manager.clear_state()
```

### State Size

Typical state file: **5-10 MB**
- Small compared to model weights (50-300 MB)
- Committed with each cycle
- Git LFS not needed (under 10 MB)

## Integration with Adaptive Parameter Budget

Sample count is preserved across cycles:

```python
total_samples_seen = 500,000  # Accumulated over 50 cycles
parameter_budget = calculate_budget(total_samples_seen)
```

This ensures:
- Budget grows correctly as training progresses
- Model can scale architecture when justified by data
- No artificial resets that would prevent growth

## Logging

State saves are logged:

```
Epoch 2900 | Loss: 8.1234 | ... | Time: 5400s | Remaining: 12600s
  → Saved training state (epoch 2900, 232,000 samples)
```

## Failure Modes

### Corrupted State File

If state file is corrupted:
- System automatically starts fresh
- Logs warning
- No training interruption

### Missing State File

If state file deleted:
- Treats as first run
- Starts from epoch 0
- Initializes optimizer fresh

### Version Mismatch

If model architecture changed:
- May need to clear state
- System will attempt to load but may fail
- Clear state manually if architecture evolution breaks compatibility

## Best Practices

1. **Always use `--resume`** in automated workflows
2. **Commit state files** after each cycle
3. **Don't manually edit** state files
4. **Clear state** only when intentionally resetting
5. **Monitor sample count** to track true training progress

## Future Enhancements

- **Multi-version state**: Keep last N states for rollback
- **State compression**: Reduce file size
- **Cloud backup**: Automatically backup state to external storage
- **State visualization**: Dashboard showing training progression
