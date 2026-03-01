# AutoEvolve-ProteinFold

**Self-improving protein folding model with automated training, evolutionary architecture search, and continuous weight optimization.**

[![Continuous Training](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/actions/workflows/continuous_training_gated.yml/badge.svg)](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/actions/workflows/continuous_training_gated.yml)
[![Quality Gates](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/actions/workflows/tests.yml/badge.svg)](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/actions/workflows/tests.yml)
[![Weekly Benchmark](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/actions/workflows/benchmark.yml/badge.svg)](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/actions/workflows/benchmark.yml)

## 🚀 Overview

AutoEvolve-ProteinFold is a novel approach to protein structure prediction that continuously improves itself through:

- **Automated Training Cycles**: GitHub Actions runs training every 6 hours (5h 40m per cycle)
- **Evolutionary Architecture Search**: Model architecture evolves based on performance
- **Adaptive Parameter Budget**: Model size scales with training data to prevent overfitting
- **Quality Gates**: Comprehensive testing ensures only improvements are committed
- **Self-Modifying Weights**: Model saves improved weights back to the repository
- **Google Colab Integration**: Extended training sessions with GPU acceleration
- **Real & Synthetic Data**: Fetches from UniProt/AlphaFold APIs (no keys needed!) + generates realistic synthetic data
- **Continuous Benchmarking**: Weekly comparison against AlphaFold2, ESMFold, RoseTTAFold

## 🧬 Key Features

### Self-Improvement Loop with Safety
```
GitHub Actions → Train Model → Evaluate Performance → 
Quality Gates (must pass) → Evolve Architecture → 
Safety Checks → Save Weights → Push to GitHub → Repeat
```

If quality gates fail, changes are automatically rolled back.

### Adaptive Parameter Budget

**Prevents overfitting** by tying model size to training data:

| Training Samples | Parameter Budget | Status |
|-----------------|------------------|--------|
| 5,000 | 50M | 🟢 Conservative |
| 50,000 | 100M | 🟢 Growing |
| 500,000 | 350M | 🟡 Substantial |
| 5,000,000 | 1B (max) | 🔴 Full Scale |

Model earns larger capacity by training on more data!

### Architecture Highlights
- **Evoformer-inspired blocks** with row/column attention
- **Dynamic layer generation** - model adds/removes layers based on performance
- **Pairwise feature extraction** with learnable embeddings
- **Structure module** predicting 3D coordinates + backbone angles
- **Confidence prediction** for uncertainty quantification

## 📊 Model Architecture

```python
EvolvableProteinFoldingModel(
  embedding_dim=256,
  pair_dim=128,
  n_heads=8,
  n_blocks=6 (evolvable, 2-24 range),
  dropout=0.1,
  max_params=1B (adaptive)
)
```

The model can modify its own:
- Number of Evoformer blocks
- Hidden dimensions  
- Attention heads
- Processing pipeline

## ⏱️ Workflow Duration

**GitHub Actions Free Tier**: Each workflow can run up to **6 hours**. The training workflow is configured for:
- **5 hours 40 minutes** of actual training time
- **20 minutes** buffer for setup, evaluation, evolution, and pushing
- Runs **every 6 hours** automatically
- **~57 hours of training per week** completely automated!

**No action needed from you** - it runs autonomously once enabled.

## 🔧 Setup

### Quick Start (Zero Configuration)

1. **Fork or clone this repository**
2. **Enable GitHub Actions**: Go to the [Actions tab](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/actions)
3. **That's it!** The model trains automatically every 6 hours

### Local Development
```bash
git clone https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold.git
cd AutoEvolve-ProteinFold
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Initialize training
python scripts/train_cycle.py --mode continuous --max-time 3600
```

### Google Colab Training (Optional GPU Boost)
1. Open `colab/AutoEvolve_ProteinFold_Colab.ipynb` in [Google Colab](https://colab.research.google.com/github/Tommaso-R-Marena/AutoEvolve-ProteinFold/blob/main/colab/AutoEvolve_ProteinFold_Colab.ipynb)
2. Add your GitHub Personal Access Token to Colab Secrets:
   - Click the key icon (🔑) in the left sidebar
   - Add secret: Name = `GITHUB_TOKEN`, Value = your token
   - [Create token here](https://github.com/settings/tokens) (needs `repo` scope)
3. Run all cells to train and auto-sync weights

## 🛡️ Quality Gates & Safety

The model must pass **all** these tests before committing changes:

### Architecture Constraints
- Block count: 2-24
- Parameters: Adaptive budget (50M-1B based on data)
- Embedding dim divisible by attention heads
- All configs within safe ranges

### Performance Regression
- Max 15% worse than historical best (allows exploration)
- No NaN or Inf losses
- Improvement required after 5 cycles

### Self-Modification Safety
- Cannot modify workflows, tests, or core training scripts
- No dangerous code patterns (eval, exec, os.system)
- File size limits (100MB max)
- Syntax validation for all Python files

**If any test fails**: Changes are automatically rolled back, no commit is made.

## 📈 Training Data Sources

1. **UniProt API**: Fetches reviewed protein sequences (no API key required!)
2. **AlphaFold Database**: Downloads predicted structures (public access)
3. **Synthetic Generation**: Creates physically plausible sequences with realistic:
   - Amino acid frequency distributions
   - Bond lengths (~3.8Å C-alpha distance)
   - Backbone geometry

## 🧪 Evolution Strategy

Every training cycle:
1. Generate population of 5 architectural mutants
2. Evaluate each on validation set
3. Select best performer
4. **Quality gates check**: Architecture constraints, performance regression, safety
5. If all pass → save architecture + weights
6. If any fail → rollback changes
7. Update `config/model_config.json`
8. Commit to repository

## 📊 Monitoring Performance

Check:
- `metrics/training_metrics.json` - Latest training stats (samples, loss, time)
- `metrics/evaluation_results.json` - Model performance (RMSD, confidence)
- `metrics/performance_history.json` - Historical performance tracking
- `logs/evolution_history.json` - Architecture changes over time
- `logs/quality_failures.json` - Failed quality gate attempts
- `reports/` - Benchmark comparisons
- [Actions tab](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/actions) - Live workflow runs

## 🎯 Goals

- **Short-term**: Achieve competitive performance on CASP15 benchmark
- **Mid-term**: Outperform AlphaFold2 on specific protein families
- **Long-term**: Discover novel architecture patterns through evolution

## 🔬 Research Applications

- Drug discovery (predicting protein-ligand binding)
- Protein engineering (stability optimization)
- De novo protein design
- Understanding protein evolution

## 📝 Configuration

Edit `config/model_config.json` to adjust:
- Model dimensions
- Number of layers
- Training hyperparameters
- Evolution parameters

Changes are validated by quality gates before acceptance.

## 🤝 Contributing

Contributions welcome! The model improves itself, but human insights accelerate progress.

Areas for contribution:
- Better loss functions
- Advanced data augmentation
- Novel architecture components
- Benchmark dataset integration
- Improved quality gates

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- AlphaFold team (DeepMind) for revolutionizing protein folding
- UniProt for comprehensive protein database
- ESMFold and RoseTTAFold teams for open-source models

## 📧 Contact

Tommaso R. Marena - [@Tommaso-R-Marena](https://github.com/Tommaso-R-Marena)

---

**Note**: This is an experimental research project. Model performance will improve over time as it trains and evolves.

**Current Status:**
- Generation: `0`
- Parameter Budget: Adaptive (50M-1B)
- Training: Automated every 6 hours
- Quality Gates: ✅ Active

Last Updated: Auto-generated by training pipeline
