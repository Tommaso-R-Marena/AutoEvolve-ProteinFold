"""Integration test for full training-eval-evolve pipeline."""
import pytest
import torch
import json
import tempfile
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from model.architecture import EvolvableProteinFoldingModel
from scripts.evolutionary_improvements import apply_evolutionary_improvement
from scripts.structure_validation import StructureValidator
from scripts.benchmark_suite import ProteinFoldingBenchmark


class TestFullPipeline:
    def test_train_save_load_evolve(self):
        """Test complete pipeline: train -> save -> load -> evolve."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create model
            config_path = Path(__file__).parent.parent.parent / 'config' / 'model_config.json'
            with open(config_path) as f:
                config = json.load(f)
            
            model = EvolvableProteinFoldingModel(config)
            
            # Train for one step
            sequences = torch.randint(0, 20, (2, 30))
            outputs = model(sequences)
            
            # Save checkpoint
            checkpoint_path = tmpdir / 'test.pt'
            model.save_checkpoint(str(checkpoint_path))
            
            # Load checkpoint
            loaded_model = EvolvableProteinFoldingModel.load_checkpoint(str(checkpoint_path))
            
            # Evolve architecture
            loaded_model = apply_evolutionary_improvement(loaded_model, 'auxiliary_heads')
            
            # Verify model still works
            outputs2 = loaded_model(sequences)
            assert outputs2['coordinates'].shape == outputs['coordinates'].shape
    
    def test_validation_after_training(self):
        """Test structure validation on model outputs."""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'model_config.json'
        with open(config_path) as f:
            config = json.load(f)
        
        model = EvolvableProteinFoldingModel(config)
        sequences = torch.randint(0, 20, (1, 50))
        outputs = model(sequences)
        
        # Validate structure
        validator = StructureValidator()
        results = validator.validate_structure(outputs['coordinates'][0])
        
        assert 'validity_score' in results
        assert 0 <= results['validity_score'] <= 1
    
    def test_benchmark_computation(self):
        """Test benchmark metrics can be computed."""
        benchmark = ProteinFoldingBenchmark()
        
        pred_coords = torch.randn(50, 3)
        true_coords = torch.randn(50, 3)
        
        tm_score = benchmark.compute_tm_score(pred_coords, true_coords)
        lddt = benchmark.compute_lddt(pred_coords, true_coords)
        rmsd = benchmark.compute_rmsd(pred_coords, true_coords)
        
        assert 0 <= tm_score <= 1
        assert 0 <= lddt <= 1
        assert rmsd >= 0
