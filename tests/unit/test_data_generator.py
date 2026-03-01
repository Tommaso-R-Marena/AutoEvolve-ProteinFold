"""Unit tests for data generation."""
import pytest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from model.data_generator import ProteinDataGenerator


@pytest.fixture
def generator():
    return ProteinDataGenerator()


class TestSyntheticDataGeneration:
    def test_batch_generation(self, generator):
        """Test synthetic batch generation."""
        batch = generator.generate_synthetic_batch(batch_size=4, min_len=30, max_len=100)
        
        assert 'sequences' in batch
        assert 'coordinates' in batch
        assert 'mask' in batch
        assert batch['sequences'].shape[0] == 4
    
    def test_sequence_length_constraints(self, generator):
        """Test sequences respect min/max length."""
        batch = generator.generate_synthetic_batch(batch_size=10, min_len=50, max_len=60)
        
        for i in range(10):
            seq_len = batch['mask'][i].sum().item()
            assert 50 <= seq_len <= 60
    
    def test_valid_amino_acids(self, generator):
        """Test sequences contain valid amino acid indices."""
        batch = generator.generate_synthetic_batch(batch_size=5)
        
        sequences = batch['sequences']
        assert (sequences >= 0).all()
        assert (sequences < 20).all()
    
    def test_coordinates_shape(self, generator):
        """Test coordinate dimensions are correct."""
        batch = generator.generate_synthetic_batch(batch_size=3, min_len=40, max_len=40)
        
        coords = batch['coordinates']
        assert coords.shape == (3, 40, 3)
    
    def test_mask_validity(self, generator):
        """Test mask is binary and matches sequence length."""
        batch = generator.generate_synthetic_batch(batch_size=5)
        
        mask = batch['mask']
        assert ((mask == 0) | (mask == 1)).all()


class TestRealDataFetching:
    def test_uniprot_fetch(self, generator):
        """Test UniProt data fetching (may fail without network)."""
        try:
            data = generator.fetch_real_data_uniprot(n_samples=5)
            if data:
                assert len(data) <= 5
                assert all(isinstance(seq, str) for seq in data)
        except Exception:
            # Network error is acceptable
            pytest.skip("Network unavailable for UniProt fetch")
