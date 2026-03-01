# AutoEvolve-ProteinFold

**Self-Evolving AI System for Protein Structure Prediction**

A revolutionary protein folding model that automatically improves its own code, architecture, and training procedures through continuous self-modification and quality-gated evolution.

[![Gated Continuous Training](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/actions/workflows/continuous_training_gated.yml/badge.svg)](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/actions/workflows/continuous_training_gated.yml)

## 🚀 Key Features

### Self-Modifying Architecture
- **Dynamic layer generation**: Model adds/removes layers based on performance
- **Adaptive hyperparameters**: Learning rate, batch size, and architecture evolve
- **Quality-gated evolution**: Changes only committed if they pass rigorous tests

### Continuous Training & Evolution
- **24/7 autonomous training**: Runs on GitHub Actions every 6 hours
- **Seamless resumption**: Training state preserved across runs
- **Real protein data**: Fetches sequences from UniProt automatically
- **Memory-efficient**: Optimized for GitHub Actions' 7GB RAM limit

### Advanced Visualizations
- **PyMOL publication-quality renders**: High-res confidence-colored structures
- **Interactive 3D viewers**: Web-based exploration with py3Dmol
- **Dynamic folding animations**: Watch proteins fold in real-time with confidence-driven kinetics
- **Confidence coloring**: Red (low) → Yellow (medium) → Green (high)

### Robust Error Handling
- **Comprehensive safety checks**: NaN/Inf detection, gradient clipping, memory monitoring
- **Automatic recovery**: Batch size reduction, fallback losses, state backups
- **Graceful degradation**: Continues training even with partial failures

## 📊 Current Status

- **Generation**: 0 (baseline)
- **Architecture**: 128-dim embeddings, 64-dim pairs, 2 Evoformer blocks
- **Training**: Continuous 5-hour cycles every 6 hours
- **Memory usage**: ~600 MB (comfortably under 7 GB limit)

## 🏗️ Architecture

### Model Design
```
Input Sequence
    ↓
Amino Acid Embeddings (128-dim)
    ↓
Pairwise Features (64-dim, memory-efficient)
    ↓
Evoformer Blocks (2x, row-wise attention)
    ↓
Structure Module
    ↓
Outputs:
  - 3D Coordinates
  - Backbone Angles (φ, ψ, ω)
  - Confidence Scores (per-residue)
```

### Memory Optimization
- **Row-wise pair attention**: O(L²) instead of O(L⁴)
- **Reduced dimensions**: 128 embedding, 64 pair
- **Batch size 4**: Halves memory vs default 8
- **Sequence length cap**: 200 residues max
- **Result**: 46× memory reduction (27.6 GB → 0.6 GB)

## 🎯 Training Pipeline

### Continuous Improvement Loop
1. **Train**: 5-hour training cycles with real + synthetic data
2. **Evaluate**: Quality gates check for improvements
3. **Evolve**: Architecture mutations if performance improves
4. **Test**: Regression tests ensure no degradation
5. **Commit**: Auto-commit weights, metrics, and improvements

### Quality Gates
- Loss improvement threshold: 5%
- No NaN/Inf values in predictions
- Gradient norms < 100
- Memory usage < 5 GB

## 📈 Evolution Strategy

The model earns larger capacity through proven performance:

- **Gen 0-10**: 128-dim, 2 blocks (proof of concept)
- **Gen 10-50**: 192-dim, 4 blocks (if consistently improving)
- **Gen 50+**: 256-dim, 6 blocks (if exceptional)

## 🎨 Visualization Features

### Generate Visualizations
```bash
# Publication-quality renders
python scripts/visualize_predictions.py \
  --checkpoint weights/latest.pt \
  --sequence "MKTAYIAKQRQISFVK..." \
  --multiple-views

# Interactive 3D viewer
python scripts/visualize_predictions.py \
  --checkpoint weights/latest.pt \
  --web

# Folding animation
python scripts/animate_folding.py \
  --checkpoint weights/latest.pt \
  --n-steps 60 \
  --fps 15
```

### Output Files
- `visualizations/structure_cartoon.png` - High-res rendering
- `visualizations/predicted_structure.pdb` - Structure file
- `visualizations/structure_interactive.html` - Web viewer
- `animations/folding_animation.gif` - Folding movie

## 🚦 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Manual Training Run
```bash
# Single training cycle (5 hours)
python scripts/train_cycle.py \
  --mode continuous \
  --max-time 18000 \
  --batch-size 4

# Resume from saved state
python scripts/train_cycle.py \
  --mode continuous \
  --max-time 18000 \
  --resume
```

### Trigger Automated Training
1. Go to [Actions tab](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/actions/workflows/continuous_training_gated.yml)
2. Click "Run workflow"
3. Select branch: `main`
4. Click "Run workflow" button

## 📊 Monitoring

### View Training Progress
- **Logs**: Check [Actions runs](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/actions)
- **Metrics**: See `metrics/training_metrics.json`
- **Checkpoints**: Download from `weights/latest.pt`
- **Artifacts**: Training logs uploaded to GitHub Artifacts

### Key Metrics
```json
{
  "generation": 0,
  "epochs": 1000,
  "final_loss": 8.234,
  "best_loss": 7.891,
  "training_time": 18000,
  "total_samples": 4000,
  "completed_successfully": true
}
```

## 🛡️ Error Handling

The system handles:
- ✅ Memory allocation errors → Reduce batch size
- ✅ NaN/Inf in predictions → Skip batch, continue
- ✅ Gradient explosions → Clip and warn
- ✅ Corrupted checkpoints → Load from backup
- ✅ Network failures → Use cached data
- ✅ 10+ consecutive errors → Save state and stop

## 📁 Repository Structure

```
AutoEvolve-ProteinFold/
├── model/
│   ├── architecture.py          # Self-modifying model
│   └── data_generator.py        # Synthetic + real data
├── scripts/
│   ├── train_cycle.py          # Training loop
│   ├── evaluate.py             # Performance evaluation
│   ├── evolve_architecture.py  # Genetic algorithm
│   ├── visualize_predictions.py # PyMOL renders
│   └── animate_folding.py      # Folding animations
├── tests/
│   ├── test_architecture_constraints.py
│   └── test_performance_regression.py
├── .github/workflows/
│   └── continuous_training_gated.yml
├── config/
│   └── model_config.json       # Hyperparameters
├── data/
│   ├── protein_database/       # UniProt sequences
│   └── training_state/         # Resumption state
├── weights/                     # Model checkpoints
├── metrics/                     # Training metrics
├── logs/                        # Training logs
└── visualizations/              # Generated images
```

## 🔬 Research Goals

### Short-term (Generations 0-10)
- ✅ Stable training pipeline
- ✅ Memory-efficient architecture
- ⏳ Achieve loss < 5.0
- ⏳ Generate valid protein structures

### Medium-term (Generations 10-50)
- ⏳ Match ESMFold on small proteins
- ⏳ Expand to 256-dim embeddings
- ⏳ Add multiple sequence alignment (MSA) features
- ⏳ Improve confidence prediction

### Long-term (Generations 50+)
- ⏳ Compete with AlphaFold2 on benchmarks
- ⏳ Novel architecture discoveries
- ⏳ Publish peer-reviewed results
- ⏳ Open-source trained models

## 🤝 Contributing

This is a research project in active development. The model evolves itself, but human guidance is welcome:

- **Bug reports**: Open an issue
- **Feature ideas**: Start a discussion
- **Code improvements**: Submit a PR (competes with auto-generated improvements!)

## 📜 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **AlphaFold2**: Architecture inspiration
- **ESMFold**: Proving smaller models can work
- **UniProt**: Real protein sequence data
- **GitHub Actions**: Free compute for continuous training

## 📧 Contact

Tommaso Marena - [@Tommaso-R-Marena](https://github.com/Tommaso-R-Marena)

Project Link: [https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold)

---

**Status**: 🟢 Active Development | **Last Updated**: March 2026 | **Model Generation**: 0
