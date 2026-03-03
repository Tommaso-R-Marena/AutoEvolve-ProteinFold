# Reproducibility Guide

## Complete Instructions for Reproducing Results

This document provides everything needed to reproduce our results from scratch.

## Environment Setup

### Prerequisites
```bash
# Python 3.10+
python --version

# Git
git --version
```

### Installation

```bash
# Clone repository
git clone https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold.git
cd AutoEvolve-ProteinFold

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Hardware Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4 GB
- Disk: 5 GB

**Recommended:**
- CPU: 4+ cores
- RAM: 16 GB
- GPU: NVIDIA GPU with 8GB+ VRAM (optional, speeds up training 10x)
- Disk: 20 GB (for caching structures)

## Data Collection

### Automated (Recommended)

```bash
# Fetch 50 fresh proteins from UniProt + AlphaFold
python scripts/train_revolutionary.py \
  --n-fresh-samples 50 \
  --max-time 100  # Just test data fetch
```

This will:
1. Query UniProt for 50 diverse sequences
2. Download AlphaFold structures
3. Cache locally in `data/protein_cache/`
4. Generate metadata

### Manual

If automated fetching fails:

1. **Download sequences from UniProt:**
   - Go to https://www.uniprot.org/
   - Search for reviewed proteins
   - Download FASTA format
   - Place in `data/sequences/`

2. **Download structures from AlphaFold DB:**
   - Visit https://alphafold.ebi.ac.uk/
   - Search by UniProt ID
   - Download PDB files
   - Convert to numpy arrays using our converter

## Training Runs

### Quick Test (5 minutes)

```bash
python scripts/train_revolutionary.py \
  --max-time 300 \
  --batch-size 2
```

### Standard Training (5 hours)

```bash
python scripts/train_revolutionary.py \
  --max-time 18000 \
  --batch-size 4 \
  --n-fresh-samples 50
```

### With Neural Architecture Search

```bash
python scripts/train_revolutionary.py \
  --max-time 18000 \
  --batch-size 4 \
  --use-nas \
  --n-fresh-samples 50
```

### Resume Training

Training automatically resumes from checkpoints:

```bash
python scripts/train_revolutionary.py \
  --max-time 18000 \
  # Automatically loads from weights/latest.pt
```

## Evaluation

### Benchmark on Test Set

```python
from scripts.comprehensive_benchmark import BenchmarkSuite
from model.enhanced_architecture import RevolutionaryProteinFolder
import torch

# Load model
model = RevolutionaryProteinFolder.load_checkpoint('weights/latest.pt')

# Prepare test set (your data)
test_set = [...]  # List of (sequence, true_coords, protein_id)

# Run benchmark
benchmark = BenchmarkSuite(model, device='cuda')
results = benchmark.run_benchmark_suite(test_set)
benchmark.save_results('results/benchmark.json')
```

### Generate Predictions

```python
import torch
from model.enhanced_architecture import RevolutionaryProteinFolder

model = RevolutionaryProteinFolder.load_checkpoint('weights/latest.pt')
model.eval()

# Your sequence (as tensor of amino acid indices)
sequence = torch.randint(0, 20, (1, 100))

with torch.no_grad():
    predictions = model(sequence, num_recycles=3)

coords = predictions['coordinates'][0].numpy()  # [L, 3]
confidence = predictions['confidence'][0].numpy()  # [L]
```

## Random Seeds

### For Exact Reproducibility

```python
import torch
import numpy as np
import random

# Set all seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# For CUDA determinism
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

## Expected Results

### Generation 0 (Baseline)

```
Metrics after 500 epochs (~5 hours):
  RMSD: 15-20 Å
  TM-score: 0.3-0.4
  lDDT: 40-50
  Training time: ~5 hours
```

### Generation 10 (Target)

```
Metrics after ~50 training cycles:
  RMSD: 5-8 Å
  TM-score: 0.5-0.6
  lDDT: 60-70
  Total training: ~250 hours
```

## Checkpoints

All checkpoints are saved to `weights/` with metadata:

```json
{
  "epoch": 500,
  "best_loss": 8.234,
  "timestamp": 1709452800,
  "generation": 0,
  "config": {...}
}
```

## Logs

Training logs saved to:
- `logs/training_{timestamp}.log`
- `metrics/training_metrics.json`
- `metrics/revolutionary_training.json`

## Common Issues

### Out of Memory

```bash
# Reduce batch size
python scripts/train_revolutionary.py --batch-size 2

# Or reduce max sequence length in config.json:
{"max_sequence_length": 128}
```

### Network Errors (AlphaFold fetch)

```bash
# Use synthetic data only
python scripts/train_cycle.py --synthetic-only
```

### Slow Training

```bash
# Reduce recycling iterations
# Edit config.json: {"num_recycles": 1}
```

## Docker (Optional)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "scripts/train_revolutionary.py"]
```

Build and run:
```bash
docker build -t autoevolve .
docker run -v $(pwd)/weights:/app/weights autoevolve
```

## Citation

If you use this code:

```bibtex
@software{autoevolve_2026,
  author = {Marena, Tommaso R.},
  title = {AutoEvolve-ProteinFold: Self-Evolving AI for Protein Structure Prediction},
  year = {2026},
  url = {https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold},
  note = {Commit: [INSERT COMMIT SHA]}
}
```

## Questions?

Open an issue: https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/issues
