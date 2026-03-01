# Adaptive Parameter Budget

## Motivation

Static parameter limits (e.g., "500M parameters max") are arbitrary and don't account for the amount of training data available. This can lead to:

1. **Overfitting**: Large models on small datasets memorize rather than learn
2. **Underfitting**: Artificially constraining model capacity when abundant data is available
3. **Inefficiency**: Not utilizing available capacity optimally

## Solution: Data-Proportional Parameter Budget

The model now uses an **adaptive parameter budget** that scales with training data:

```python
parameter_budget = f(training_samples)
```

### Scaling Formula

For small datasets (< 10K samples):
```
budget = samples × 5-10 params/sample
```

For larger datasets:
```
budget = base_budget × (1 + log₁₀(samples) - 3)
```

### Example Budgets

| Training Samples | Parameter Budget | Params/Sample |
|-----------------|------------------|---------------|
| 5,000 | 50M | 10,000 |
| 10,000 | 50M | 5,000 |
| 50,000 | 100M | 2,000 |
| 100,000 | 150M | 1,500 |
| 500,000 | 350M | 700 |
| 1,000,000 | 500M | 500 |
| 5,000,000 | 1B (max) | 200 |

## Implementation Details

### Data Size Estimation

The system estimates training data from:

1. **Training logs**: `epochs × samples_per_epoch`
2. **Downloaded data**: Files in `data/benchmark/` directory
3. **Real sequences**: UniProt fetches tracked in `data/training_data_info.json`
4. **Synthetic data**: Estimated from training metrics

### Quality Gates

The architecture constraint tests now check:

```python
n_samples = estimate_training_samples()
budget = calculate_parameter_budget(n_samples)

assert model_params <= budget
```

If the model grows too large for the available data, the quality gates fail and changes are rolled back.

### Warning Threshold

The system warns when `params_per_sample > 20`, indicating potential overfitting risk even within budget.

## Benefits

1. **Prevents overfitting**: Model size naturally constrained by data availability
2. **Enables scaling**: As the model fetches more data (UniProt, AlphaFold), it can grow
3. **Self-regulating**: No manual intervention needed - budget adjusts automatically
4. **Scientifically grounded**: Based on statistical learning theory (VC dimension, sample complexity)

## Future Enhancements

- **Cross-validation**: Use held-out set to empirically detect overfitting
- **Per-layer budgets**: Allocate parameters differently for embeddings vs. attention
- **Data quality weighting**: Real protein structures count more than synthetic
- **Active learning**: Prioritize fetching data where model is uncertain

## Monitoring

Check parameter utilization in test output:

```
Adaptive Parameter Budget:
  Training samples (estimated): 25,000
  Current parameters: 45,234,567 (45.2M)
  Parameter budget: 75,000,000 (75.0M)
  Params per sample: 1,809.4
  Utilization: 60.3%
```

This shows the model is using 60% of its data-justified capacity - room to grow!
