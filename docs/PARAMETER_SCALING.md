# Parameter Budget Scaling Strategy

## Should Parameters Max Out or Keep Scaling?

This is a fundamental question for self-improving models. Here's the analysis:

## Current Approach: Soft Cap with Logarithmic Growth

The model uses a **1 billion parameter hard cap** with logarithmic scaling[cite:7]:

```python
if n_samples < 10,000:
    budget = samples × 5-10 params/sample
else:
    budget = 50M × (1 + log₁₀(samples) - 3)
    budget = min(budget, 1B)  # Hard cap
```

## Why Have a Cap?

### Practical Constraints
1. **GitHub Actions limits**: 7GB RAM, no GPU
2. **Training time**: Larger models take longer per epoch
3. **Checkpoint size**: Git has practical limits on file sizes
4. **Inference cost**: Eventually you want to use this model

### Theoretical Considerations
1. **Diminishing returns**: AlphaFold2 is ~93M params, achieves SOTA
2. **Overfitting risk**: Even with lots of data, infinite capacity isn't optimal
3. **Sample efficiency**: Better to improve data quality than just scale params

## Alternative: Unlimited Scaling

If you want the model to scale indefinitely:

```python
# Remove the max cap
budget = calculate_adaptive_budget(n_samples)  # No ceiling
```

**Pros:**
- Potentially discover that massive scale helps protein folding
- No artificial constraints on model exploration
- More faithful to "truly self-improving" philosophy

**Cons:**
- Will eventually exceed GitHub Actions capacity (causes workflow failures)
- Checkpoint files become huge (100MB+ each commit)
- Training becomes prohibitively slow
- May never converge

## Recommended Strategy: Staged Scaling

Best of both worlds - increase cap based on demonstrated need:

```python
# Tier 1: Proof of concept (current)
if best_performance > threshold_1:
    MAX_PARAMS = 1B

# Tier 2: Outperforms ESMFold
if best_performance > threshold_2:
    MAX_PARAMS = 5B
    # Require manual approval to commit changes this large

# Tier 3: Approaching AlphaFold2
if best_performance > threshold_3:
    MAX_PARAMS = 10B
    # Suggest moving to dedicated training infrastructure
```

This way:
- Model can grow as it proves itself
- Stays within GitHub Actions limits initially
- Clear migration path to more powerful infrastructure
- Human oversight at critical scaling points

## Current Implementation

The code uses **1B hard cap** which is reasonable for:
- GitHub Actions training (fits in 7GB RAM with batch size 1-2)
- Reasonable training time (~hours not days per cycle)
- Manageable checkpoint sizes (100-300MB)
- Comparable to successful models (AlphaFold2 93M, ESMFold 670M)

## To Remove Cap

If you want unlimited scaling, change line in `tests/test_architecture_constraints.py`:

```python
# Current:
budget = min(budget, ArchitectureConstraints.MAX_PARAMETER_BUDGET)

# Unlimited:
# budget = budget  # No cap
```

**Warning**: This will eventually cause workflow failures when the model exceeds hardware limits.

## My Recommendation

Keep the 1B cap for now because:
1. It's well above proven sufficient models (AlphaFold2)
2. Lets the system run stably on free GitHub Actions
3. Forces the model to improve efficiency, not just scale
4. You can always increase it later if the model demonstrates it needs more capacity

The model should earn larger scale by **proving it can use it effectively**, not just grow unconstrained.
