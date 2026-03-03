# AutoEvolve-ProteinFold

**Revolutionary Self-Evolving AI for Protein Structure Prediction**

A groundbreaking protein folding system that combines state-of-the-art deep learning techniques with autonomous self-improvement. This system doesn't just predict protein structures—it evolves its own architecture, learns from real AlphaFold data, and provides calibrated uncertainty estimates.

[![Gated Continuous Training](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/actions/workflows/continuous_training_gated.yml/badge.svg)](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/actions/workflows/continuous_training_gated.yml)

## 🌟 What Makes This Revolutionary?

### 🤖 Self-Evolving Architecture
- **Neural Architecture Search (NAS)**: Automatically discovers optimal layer types and connections
- **Dynamic layer generation**: Model grows/shrinks based on performance
- **Learned hyperparameters**: Architecture parameters optimized alongside weights
- **Generation-based evolution**: Each generation builds on previous successes

### 🧬 Real Data Integration
- **AlphaFold Database**: Fetches and trains on real protein structures
- **UniProt sequences**: 50+ real sequences per training cycle
- **Structure caching**: Persistent storage for instant reuse
- **30/70 mixing**: Optimal blend of real AlphaFold + synthetic data

### 🎯 Uncertainty Quantification
- **Ensemble predictions**: Multiple models for robust estimates
- **MC Dropout**: Epistemic uncertainty during inference
- **Evidential learning**: Separates aleatoric vs epistemic uncertainty
- **Conformal prediction**: Statistically valid prediction intervals

### 🔬 Advanced Architecture

#### Diffusion-Based Generation
- **Iterative refinement**: 20-step denoising process like AlphaFold3
- **High-quality structures**: Better geometry than direct prediction
- **Guidance sampling**: Controllable generation process

#### Invariant Point Attention (IPA)
- **SE(3)-equivariant**: Respects 3D symmetries
- **Operates on coordinates**: Direct manipulation of 3D space
- **From AlphaFold2**: Proven state-of-the-art technique

#### Geometric Constraints
- **Chirality preservation**: Enforces L-amino acid stereochemistry
- **Distance constraints**: Maintains realistic peptide bond lengths (~3.8Å)
- **Physical plausibility**: Hard-coded chemistry knowledge

#### Rotary Position Encoding (RoPE)
- **Modern embeddings**: Superior to learned positional encodings
- **From GPT architectures**: Proven at scale

### 📊 Comprehensive Benchmarking
- **RMSD**: Root mean square deviation
- **TM-score**: Template modeling score (0-1, >0.5 = good)
- **GDT-TS/HA**: Global distance tests
- **lDDT**: Local distance difference test
- **Speed metrics**: Residues/sec, proteins/sec

### ⚙️ Autonomous Training
- **24/7 GitHub Actions**: Trains every 6 hours automatically
- **Seamless resumption**: Picks up exactly where it left off
- **Quality gates**: Only commits improvements
- **Memory optimized**: Fits in 7GB RAM limit

## 📊 Current Status

**Model Capabilities:**
- ✅ Stable training on AlphaFold ground truth
- ✅ Diffusion-based structure generation
- ✅ SE(3)-equivariant geometry processing
- ✅ Calibrated uncertainty estimates
- ✅ Multi-scale recycling (3 iterations)
- ✅ Physical constraint satisfaction

**Architecture:**
- **Embedding dim**: 128 (expandable to 256)
- **Pair dim**: 64 (memory-efficient)
- **Evoformer blocks**: 2 (grows with performance)
- **IPA blocks**: 3 (invariant point attention)
- **Diffusion steps**: 20 (structure refinement)
- **Generation**: 0 (baseline, will evolve)

## 🏛️ Architecture Overview

### Revolutionary Combined Architecture

```
Input: Amino Acid Sequence
         ↓
   Embeddings (128-dim) + Rotary Position Encoding
         ↓
   Pairwise Features (64-dim, outer product)
         ↓
   ┌────────────────────────────┐
   │   Recycling Loop (3x)     │
   │   │                          │
   │   └──> Evoformer Blocks (2x)  │
   │        ↓                      │
   │   Invariant Point Attention  │
   │        ↓                      │
   │   Geometric Features         │
   │        ↓                      │
   │   Structure Update           │
   │        ↓                      │
   │   Physical Constraints       │
   └────────────────────────────┘
         ↓
   Diffusion Refinement (20 steps)
         ↓
Outputs:
  • 3D Coordinates (CA atoms)
  • Backbone Angles (φ, ψ, ω)
  • Per-residue Confidence
  • Uncertainty Estimates
```

### Neural Architecture Search (Optional)

```
Searchable Operations:
  • Multi-head Attention
  • 1D Convolution
  • Gated FFN (SwiGLU)
  • Identity/Skip
         ↓
  Architecture Parameters (learned)
         ↓
  Weighted Combination
         ↓
  Best Operation Selected
```

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold.git
cd AutoEvolve-ProteinFold
pip install -r requirements.txt
```

### Quick Start

**Run basic training:**
```bash
python scripts/train_cycle.py \
  --max-time 18000 \
  --batch-size 4
```

**Use enhanced architecture:**
```python
from model.enhanced_architecture import RevolutionaryProteinFolder
import torch

config = {
    'vocab_size': 20,
    'embedding_dim': 128,
    'pair_dim': 64,
    'n_heads': 8,
    'n_blocks': 2,
    'n_structure_blocks': 3,
    'dropout': 0.1,
    'use_diffusion': True
}

model = RevolutionaryProteinFolder(config)
sequence = torch.randint(0, 20, (1, 100))  # Random sequence
outputs = model(sequence, num_recycles=3)

print(f"Predicted coordinates: {outputs['coordinates'].shape}")
print(f"Confidence: {outputs['confidence'].mean():.3f}")
print(f"Epistemic uncertainty: {outputs['constraint_losses']}")
```

**Run comprehensive benchmark:**
```python
from scripts.comprehensive_benchmark import BenchmarkSuite

benchmark = BenchmarkSuite(model, device='cuda')
stats = benchmark.run_benchmark_suite(test_set)
benchmark.save_results('results/benchmark_gen0.json')
```

### Advanced Features

**Uncertainty quantification:**
```python
from model.uncertainty_quantification import MCDropoutPredictor

predictor = MCDropoutPredictor(model, n_samples=20)
results = predictor.predict(sequence)

print(f"Mean coordinates: {results['coordinates']}")
print(f"Epistemic uncertainty: {results['epistemic_uncertainty']}")
print(f"Coordinate std: {results['coordinate_std']}")
```

**Neural architecture search:**
```python
from model.neural_architecture_search import NASProteinModel, NASTrainer

nas_model = NASProteinModel(config)
trainer = NASTrainer(nas_model)

# Train with architecture search
for epoch in range(100):
    losses = trainer.train_step(train_batch, val_batch, criterion)
    
# Get learned architecture
architecture = nas_model.get_architecture_summary()
print(f"Learned architecture: {architecture}")
```

## 🧪 Benchmarking Results

### Metrics Explained

| Metric | Range | Good | Excellent | Description |
|--------|-------|------|-----------|-------------|
| **RMSD** | 0-∞ Å | <5Å | <2Å | Coordinate deviation |
| **TM-score** | 0-1 | >0.5 | >0.7 | Fold similarity |
| **GDT-TS** | 0-100 | >50 | >80 | Global accuracy |
| **lDDT** | 0-100 | >60 | >90 | Local geometry |

### Expected Performance Trajectory

```
Generation 0 (Current):
  RMSD: 15-20Å (baseline)
  TM-score: 0.3-0.4 (learning)
  lDDT: 40-50 (developing)

Generation 10 (Target):
  RMSD: 5-8Å (useful)
  TM-score: 0.5-0.6 (correct fold)
  lDDT: 60-70 (good geometry)

Generation 50 (Goal):
  RMSD: 2-4Å (excellent)
  TM-score: 0.7-0.8 (high confidence)
  lDDT: 80-90 (near-native)
```

## 🔬 Scientific Innovations

### 1. Diffusion Models for Proteins

Inspired by **RFdiffusion** and **AlphaFold3**, we use denoising diffusion for structure generation:

**Advantages:**
- Better geometry than direct regression
- Natural uncertainty quantification
- Iterative refinement process
- State-of-the-art in 2024-2026

### 2. SE(3)-Equivariant Processing

**Invariant Point Attention** from AlphaFold2:
- Respects rotational/translational symmetry
- Operates directly on 3D coordinates
- More sample-efficient than standard attention

### 3. Neural Architecture Search

**Differentiable NAS:**
- Learns optimal operations (attention vs conv vs FFN)
- Architecture parameters trained with gradient descent
- Discovers novel architectures automatically

### 4. Uncertainty-Aware Training

**Multiple uncertainty sources:**
- **Aleatoric**: Inherent data noise (irreducible)
- **Epistemic**: Model uncertainty (reducible with more data)
- **Calibrated intervals**: Statistically valid via conformal prediction

### 5. Physical Constraints

**Hard-coded chemistry:**
- Chirality: All L-amino acids (except glycine)
- Bond lengths: CA-CA distance ≈ 3.8 ± 0.1Å
- Angles: Realistic Ramachandran distributions

## 📚 Publications & References

**Core Architecture:**
- AlphaFold2 (Jumper et al., 2021) - Evoformer, IPA
- AlphaFold3 (Abramson et al., 2024) - Diffusion models
- ESMFold (Lin et al., 2023) - Efficient folding

**Techniques:**
- RFdiffusion (Watson et al., 2023) - Diffusion for proteins
- DARTS (Liu et al., 2019) - Differentiable NAS
- Evidential DL (Amini et al., 2020) - Uncertainty quantification
- Conformal Prediction (Angelopoulos & Bates, 2021) - Calibration

## 🛠️ Repository Structure

```
AutoEvolve-ProteinFold/
├── model/
│   ├── architecture.py              # Original Evoformer model
│   ├── enhanced_architecture.py     # Revolutionary combined model
│   ├── diffusion_module.py          # Diffusion-based generation
│   ├── geometric_features.py        # IPA, constraints, geometry
│   ├── neural_architecture_search.py # NAS implementation
│   ├── uncertainty_quantification.py # Uncertainty methods
│   └── data_generator.py            # AlphaFold + synthetic data
├── scripts/
│   ├── train_cycle.py              # Training with AlphaFold data
│   ├── comprehensive_benchmark.py  # Professional metrics
│   ├── evaluate.py                 # Model evaluation
│   ├── evolve_architecture.py      # Genetic algorithm
│   └── visualize_predictions.py    # PyMOL renders
├── tests/
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   └── test_performance_regression.py
├── .github/workflows/
│   └── continuous_training_gated.yml # Autonomous training
├── config/
│   └── model_config.json           # Hyperparameters
├── data/
│   ├── protein_cache/              # Cached AlphaFold structures
│   └── training_state/             # Resumption checkpoints
├── weights/                       # Model checkpoints
├── metrics/                       # Training metrics
└── results/                       # Benchmark results
```

## 🎯 Roadmap

### Phase 1: Foundation (Generations 0-10) ✅
- [x] Stable training pipeline
- [x] AlphaFold data integration
- [x] Diffusion-based generation
- [x] Uncertainty quantification
- [x] Comprehensive benchmarking
- [ ] Loss < 5.0
- [ ] TM-score > 0.5

### Phase 2: Refinement (Generations 10-30)
- [ ] Neural architecture search deployment
- [ ] Expand to 192-dim embeddings
- [ ] Add MSA (multiple sequence alignment) features
- [ ] Improve per-residue confidence
- [ ] Match ESMFold on CAMEO

### Phase 3: Excellence (Generations 30-50)
- [ ] 256-dim embeddings with 6 blocks
- [ ] Compete with AlphaFold2 on CASP benchmarks
- [ ] Novel architecture discoveries
- [ ] Sub-2Å RMSD on test sets
- [ ] TM-score > 0.8 consistently

### Phase 4: Beyond (Generations 50+)
- [ ] Protein design capabilities
- [ ] Dynamics prediction
- [ ] Protein-protein docking
- [ ] Drug binding site prediction
- [ ] Publish in Nature/Science

## 💻 Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest tests/ -v --cov=model --cov=scripts --cov-report=term
```

### Code Quality

```bash
# Format code
black model/ scripts/ tests/

# Lint
flake8 model/ scripts/ tests/

# Type checking
mypy model/ scripts/
```

## 🤝 Contributing

This is a research project pushing the boundaries of self-evolving AI. Contributions welcome:

**Ways to contribute:**
- 🐛 Report bugs or issues
- 💡 Suggest novel architecture components
- 📊 Share benchmark results
- 📖 Improve documentation
- 🧪 Add new protein datasets

**Code contributions:**
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-idea`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-idea`)
5. Open a Pull Request

Your code competes with the auto-generated improvements! 🚀

## 📜 Citation

If you use this work in your research:

```bibtex
@software{autoevolve_proteinfold_2026,
  author = {Marena, Tommaso R.},
  title = {AutoEvolve-ProteinFold: Self-Evolving AI for Protein Structure Prediction},
  year = {2026},
  url = {https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold},
  note = {Revolutionary protein folding system with autonomous architecture evolution}
}
```

## 🛡️ License

MIT License - See [LICENSE](LICENSE) file for details

## 🚀 Acknowledgments

**Scientific Inspirations:**
- DeepMind AlphaFold Team - Revolutionary protein folding
- Meta ESM Team - Efficient language models for proteins
- Baker Lab - RFdiffusion and structure design
- Yann LeCun - Deep learning foundations

**Infrastructure:**
- GitHub Actions - Free compute for continuous training
- PyTorch - Deep learning framework
- UniProt - Protein sequence database
- AlphaFold Database - Ground truth structures

**Community:**
- The Catholic University of America
- Open-source ML/Biology community

---

**Status**: 🟢 Active Development | **Last Updated**: March 2026 | **Generation**: 0 → ∞

**Built with ❤️ by [Tommaso R. Marena](https://github.com/Tommaso-R-Marena)**

*"The best way to predict protein structures is to invent them."* – Adapted from Alan Kay
