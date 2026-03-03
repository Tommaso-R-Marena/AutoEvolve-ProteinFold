# Contributing to AutoEvolve-ProteinFold

Thank you for your interest in contributing! This project pushes the boundaries of self-evolving AI for protein structure prediction.

## Ways to Contribute

### 🐛 Report Bugs

Found a bug? [Open an issue](https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold/issues/new) with:
- Clear description
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU)
- Relevant logs or error messages

### 💡 Propose Features

Have an idea for improvement? We welcome:
- Novel architecture components
- Training optimizations
- New data sources
- Benchmark improvements
- Visualization enhancements

**Before implementing:**
1. Open an issue to discuss
2. Get feedback from maintainers
3. Align with project goals

### 🧬 Add Datasets

Contribute protein structures:
- High-quality experimental structures
- Diverse protein families
- Hard cases (disordered regions, large proteins)
- Novel folds

### 📊 Share Results

Run experiments and share:
- Benchmark results
- Comparisons with other models
- Novel architectures discovered by NAS
- Training curves
- Failure cases

### 📖 Improve Documentation

- Fix typos
- Clarify instructions
- Add examples
- Write tutorials
- Translate documentation

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/AutoEvolve-ProteinFold.git
cd AutoEvolve-ProteinFold
git remote add upstream https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold.git
```

### 2. Create Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### 3. Create Branch

```bash
git checkout -b feature/amazing-feature
```

### 4. Make Changes

Follow our coding standards:

**Code Style:**
```bash
# Format code
black model/ scripts/ tests/

# Check style
flake8 model/ scripts/ tests/

# Type checking
mypy model/ scripts/
```

**Testing:**
```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=model --cov=scripts --cov-report=term

# Specific test
pytest tests/test_architecture.py -v
```

### 5. Commit

```bash
git add .
git commit -m "feat: Add amazing feature"
```

**Commit message format:**
```
type(scope): description

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- test: Tests
- refactor: Code refactoring
- perf: Performance improvement
- chore: Maintenance
```

### 6. Push and PR

```bash
git push origin feature/amazing-feature
```

Then open a Pull Request on GitHub.

## Code Standards

### Python Style

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions focused (<50 lines)
- Use descriptive variable names

**Example:**
```python
def compute_rmsd(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Compute Root Mean Square Deviation.
    
    Args:
        predicted: Predicted coordinates [L, 3]
        target: Target coordinates [L, 3]
        mask: Valid residue mask [L]
    
    Returns:
        RMSD value in Angstroms
    
    Raises:
        ValueError: If shapes don't match
    """
    if predicted.shape != target.shape:
        raise ValueError(f"Shape mismatch: {predicted.shape} vs {target.shape}")
    
    # Implementation...
    return rmsd
```

### Testing Requirements

**All new code must have tests:**

```python
import pytest
import torch
from model.architecture import EvolvableProteinFoldingModel

def test_model_forward_pass():
    """Test that model produces valid outputs."""
    config = {'vocab_size': 20, 'embedding_dim': 128, ...}
    model = EvolvableProteinFoldingModel(config)
    
    sequence = torch.randint(0, 20, (2, 50))
    outputs = model(sequence)
    
    assert outputs['coordinates'].shape == (2, 50, 3)
    assert outputs['confidence'].shape == (2, 50)
    assert not torch.isnan(outputs['coordinates']).any()
```

### Documentation

**Module docstrings:**
```python
"""Module for protein structure prediction.

This module implements the core architecture combining:
- Evoformer blocks
- Invariant Point Attention
- Diffusion-based refinement

Example:
    >>> model = RevolutionaryProteinFolder(config)
    >>> predictions = model(sequence)
"""
```

**Function docstrings:**
```python
def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Forward pass through the model.
    
    Args:
        x: Input tensor [B, L, D]
        mask: Attention mask [B, L]
    
    Returns:
        Dictionary containing:
            - coordinates: Predicted 3D coords [B, L, 3]
            - confidence: Per-residue confidence [B, L]
            - embeddings: Hidden states [B, L, D]
    
    Raises:
        RuntimeError: If input dimensions are invalid
    """
```

## Architecture Contributions

### Adding New Components

**1. Implement the module:**
```python
# model/my_new_feature.py
import torch.nn as nn

class MyNewFeature(nn.Module):
    """Novel architecture component.
    
    Based on: Paper citation here
    """
    def __init__(self, dim: int):
        super().__init__()
        # Implementation
    
    def forward(self, x):
        # Forward pass
        return x
```

**2. Add tests:**
```python
# tests/test_my_new_feature.py
def test_my_new_feature():
    feature = MyNewFeature(dim=128)
    x = torch.randn(2, 100, 128)
    out = feature(x)
    assert out.shape == x.shape
```

**3. Integrate:**
```python
# model/enhanced_architecture.py
from model.my_new_feature import MyNewFeature

class RevolutionaryProteinFolder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.my_feature = MyNewFeature(config['dim'])
```

**4. Document:**
- Add to README
- Update TECHNICAL_PAPER.md
- Include citations

## Pull Request Process

### Checklist

Before submitting, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass (`pytest tests/`)
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] Commit messages are descriptive
- [ ] No merge conflicts
- [ ] Code is formatted (`black`)
- [ ] No linting errors (`flake8`)
- [ ] Type hints added (`mypy`)

### Review Process

1. **Automated checks:** CI runs tests, linting, type checking
2. **Code review:** Maintainer reviews code quality
3. **Discussion:** Clarifications and suggestions
4. **Approval:** Once all checks pass
5. **Merge:** Squash and merge to main

### What Reviewers Look For

- **Correctness:** Does it work as intended?
- **Performance:** Any bottlenecks or inefficiencies?
- **Clarity:** Is the code readable?
- **Tests:** Are edge cases covered?
- **Documentation:** Is it well-explained?
- **Integration:** Does it fit the architecture?

## Research Contributions

### Publishing Results

If you use this codebase for research:

1. **Cite our work:**
```bibtex
@software{autoevolve_2026,
  author = {Marena, Tommaso R.},
  title = {AutoEvolve-ProteinFold},
  year = {2026},
  url = {https://github.com/Tommaso-R-Marena/AutoEvolve-ProteinFold}
}
```

2. **Share your paper:** Open an issue with link

3. **Contribute improvements:** PR your enhancements

### Benchmark Submissions

Share your benchmark results:

```json
{
  "model_version": "v0.1.0",
  "generation": 10,
  "test_set": "CAMEO",
  "metrics": {
    "rmsd": 4.2,
    "tm_score": 0.65,
    "lddt": 72.3
  },
  "hardware": "NVIDIA A100",
  "training_time": "200 hours"
}
```

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inspiring community for all.

**Be:**
- Respectful
- Constructive
- Collaborative
- Professional

**Not acceptable:**
- Harassment
- Discrimination
- Trolling
- Spam

### Getting Help

- **Questions:** Open a GitHub Discussion
- **Bugs:** Open an Issue
- **Security:** Email tommaso@example.com (replace with actual)

### Recognition

Contributors are recognized:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Invited to co-author papers (significant contributions)

## Competing with the AI

Remember: Your code competes with auto-generated improvements!

**The bar is high:**
- Model self-optimizes through NAS
- Architecture evolves automatically
- Benchmarks improve continuously

**Stand out by:**
- Novel ideas (not just parameter tweaks)
- Rigorous evaluation
- Clear documentation
- Reproducible results

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Not sure about something? Open a Discussion or Issue!

**Happy contributing! 🚀🧬**
