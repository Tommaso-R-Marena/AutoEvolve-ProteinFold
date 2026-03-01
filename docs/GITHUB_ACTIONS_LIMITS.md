# GitHub Actions Limits & Optimization

## Current Limits (2026)

### For Public Repositories (AutoEvolve-ProteinFold)

**Compute**:
- ✅ **Minutes**: UNLIMITED (free for public repos)
- ✅ **Concurrent jobs**: 20 max
- ✅ **Single job timeout**: 6 hours (360 minutes)
- ✅ **Workflow timeout**: 72 hours
- ✅ **Queue time**: 24 hours max

**Runner Specifications**:
- **CPU**: 2-core (limited)
- **RAM**: 7 GB (our model uses ~0.6 GB)
- **Storage**: 14 GB SSD (plenty for our use)

**Storage**:
- ⚠️ **Artifacts**: 500 MB total (shared across all artifacts)
- ⚠️ **LFS**: 1 GB storage, 1 GB bandwidth per month
- ⚠️ **Repository**: Soft limit 5 GB (warnings at 1 GB)

## Our Optimization Strategy

### 1. Split Into Multiple Jobs

**Before**: Single 5h 40m job ❌
**After**: 4 parallel/sequential jobs ✅

```yaml
jobs:
  train-model:        # 3h 20m - Main training
  evaluate-model:     # 30m - Benchmarking (after training)
  evolve-architecture: # 45m - Evolution (after evaluation)
  generate-visualizations: # 30m - Parallel with evaluation
```

**Benefits**:
- Each job under 360 minute limit
- Parallel execution where possible
- Better failure isolation
- Easier to debug

### 2. Reduced Training Time

**Before**: 18,000 seconds (5 hours)
**After**: 10,800 seconds (3 hours)

**Rationale**:
- Still substantial training
- Runs 4x per day = 12 hours daily training
- Safer buffer for GitHub Actions
- Faster iteration cycles

### 3. Aggressive Cleanup

**Weekly cleanup workflow**:
- Delete artifacts older than 7 days
- Delete workflow runs older than 30 days
- Keep only 3 latest checkpoints
- Keep only 5 latest visualizations
- Keep only 10 latest logs

**Keeps storage under 500 MB limit**.

### 4. Artifact Retention

```yaml
retention-days: 7  # Instead of default 90
```

**Saves**: ~85% storage over time

### 5. Dependency Caching

```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
```

**Saves**: 1-2 minutes per run, reduces bandwidth

### 6. Timeout Guards

Every step has explicit timeout:
```yaml
timeout-minutes: 10  # Fails fast if stuck
```

**Prevents**: Wasting compute on hung processes

### 7. Emergency Stop Workflow

Manual trigger to:
- Cancel all running workflows
- Create emergency stop flag
- Prevent new runs until cleared

**Use case**: Model goes rogue, costs spiraling, etc.

## Storage Budget

### Target Allocation (< 500 MB total)

```
Weights:        150 MB (3 checkpoints × 50 MB)
Visualizations:  50 MB (5 sets × 10 MB)
Logs:            20 MB (10 logs × 2 MB)
Data:           100 MB (protein database)
Code:            30 MB (source code)
Artifacts:      150 MB (recent runs)
-----------------------------------
Total:         ~500 MB ✅
```

### Monitoring

Cleanup workflow reports:
- Repository size
- Weights directory size
- Visualizations size
- Logs size

**Alert if > 450 MB**

## Compute Budget

### Daily Usage (Public Repo = Unlimited)

**4 cycles per day × 3 hours = 12 hours training/day**

```
Training:        12 hours/day
Evaluation:       2 hours/day
Evolution:        3 hours/day
Visualization:    2 hours/day
-------------------------
Total:           19 hours/day ✅
```

**All FREE for public repositories!**

### If Moved to Private Repo

**Free tier**: 2,000 minutes/month (33 hours)
**Our usage**: ~570 hours/month

**Cost**: $0.008/minute × ~34,200 minutes = **$273/month**

**Recommendation**: Keep public or use self-hosted runner

## Best Practices We Follow

### ✅ Implemented

1. **Split long jobs** into < 6 hour chunks
2. **Cache dependencies** to save time/bandwidth
3. **Set explicit timeouts** on every step
4. **Clean up artifacts** aggressively
5. **Use `[skip ci]`** to prevent loops
6. **Parallel jobs** where possible
7. **Fail fast** with `continue-on-error`
8. **Monitor storage** weekly

### ✅ Monitoring

- Storage reported in cleanup workflow
- Training success tracked in metrics
- Benchmark progress tracked over time
- Emergency stop available

## Future Optimizations

### If Hitting Limits

1. **Reduce training frequency**: 6h → 12h cycles
2. **Smaller checkpoints**: Compress weights
3. **On-demand visualizations**: Only on request
4. **External storage**: S3 for large files
5. **Self-hosted runner**: Own hardware

### If Going Private

Options:
1. **Pay for minutes**: ~$270/month
2. **Self-hosted runner**: Free compute, $0.002/min after March 2026
3. **Reduce training**: 12h/day → 4h/day (fits in free tier)
4. **Hybrid**: Critical training on Actions, bulk on local

## Comparison: Current vs Optimized

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Single job time | 340 min | 200 min | 41% faster |
| Training time | 5h | 3h | 40% shorter |
| Storage risk | High | Low | Managed |
| Failure isolation | Poor | Good | 4 jobs |
| Parallel work | None | 2 jobs | 2× faster |
| Artifact retention | 90 days | 7 days | 93% less |
| Cleanup | Manual | Automated | Weekly |
| Emergency stop | None | Yes | Safety |

## Monitoring Commands

```bash
# Check repository size
du -sh .

# Check LFS usage
git lfs ls-files --size

# Check artifact storage (via GitHub UI)
https://github.com/<user>/<repo>/actions/caches

# Check workflow minutes (via GitHub UI)
https://github.com/<user>/<repo>/settings/billing
```

## Conclusion

✅ **Well within limits** for public repository
✅ **Optimized for efficiency** (split jobs, caching)
✅ **Aggressive cleanup** keeps storage manageable
✅ **Emergency controls** if issues arise
✅ **Scales to 12h/day training** for free

**Status**: Ready for continuous autonomous evolution! 🚀
