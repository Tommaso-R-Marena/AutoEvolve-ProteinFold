# AutoEvolve-ProteinFold

**Self-improving protein folding model with automated training, evolutionary architecture search, and continuous weight optimization.**

[![Continuous Training](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/actions/workflows/continuous_training.yml/badge.svg)](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/actions/workflows/continuous_training.yml)
[![Weekly Benchmark](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/actions/workflows/benchmark.yml/badge.svg)](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/actions/workflows/benchmark.yml)

## 🚀 Overview

AutoEvolve-ProteinFold is a novel approach to protein structure prediction that continuously improves itself through:

- **Automated Training Cycles**: GitHub Actions runs training every 6 hours
- **Evolutionary Architecture Search**: Model architecture evolves based on performance
- **Self-Modifying Weights**: Model saves improved weights back to the repository
- **Google Colab Integration**: Extended training sessions with GPU acceleration
- **Real & Synthetic Data**: Fetches from UniProt/AlphaFold APIs + generates realistic synthetic data
- **Continuous Benchmarking**: Weekly comparison against AlphaFold2, ESMFold, RoseTTAFold

## 🧬 Key Features

### Self-Improvement Loop
```
GitHub Actions → Train Model → Evaluate Performance → 
Evolve Architecture → Save Weights → Push to GitHub → Repeat
```

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
  n_blocks=6 (evolvable),
  dropout=0.1
)
```

The model can modify its own:
- Number of Evoformer blocks
- Hidden dimensions
- Attention heads
- Processing pipeline

## 🔧 Setup

### Local Development
```bash
git clone https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold.git
cd AutoEvolve-ProteinFold
pip install -r requirements.txt

# Initialize training
python scripts/train_cycle.py --mode continuous --max-time 3600
```

### Google Colab Training
1. Open `colab/AutoEvolve_ProteinFold_Colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Add your GitHub Personal Access Token to Colab Secrets (name: `GITHUB_TOKEN`)
3. Run all cells to train and auto-sync weights

### GitHub Actions (Automated)

The repository automatically trains every 6 hours via GitHub Actions. No manual intervention required!

Workflows:
- `continuous_training.yml`: Trains model, evolves architecture, commits improvements
- `colab_sync.yml`: Syncs weights from Colab training sessions
- `benchmark.yml`: Weekly performance evaluation against SOTA models

## 📈 Training Data Sources

1. **UniProt API**: Fetches reviewed protein sequences
2. **AlphaFold Database**: Downloads predicted structures
3. **Synthetic Generation**: Creates physically plausible sequences with realistic:
   - Amino acid frequency distributions
   - Bond lengths (~3.8Å C-alpha distance)
   - Backbone geometry

## 🧪 Evolution Strategy

Every training cycle:
1. Generate population of 5 architectural mutants
2. Evaluate each on validation set
3. Select best performer
4. If improvement → save architecture + weights
5. Update `config/model_config.json`
6. Commit to repository

## 📊 Monitoring Performance

Check:
- `metrics/training_metrics.json` - Latest training stats
- `metrics/evaluation_results.json` - Model performance
- `logs/evolution_history.json` - Architecture changes over time
- `reports/` - Benchmark comparisons

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

## 🤝 Contributing

Contributions welcome! The model improves itself, but human insights accelerate progress.

Areas for contribution:
- Better loss functions
- Advanced data augmentation
- Novel architecture components
- Benchmark dataset integration

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

Current Generation: `0`

Last Updated: Auto-generated by training pipeline
