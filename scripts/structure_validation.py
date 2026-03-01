#!/usr/bin/env python3
"""Validate predicted structures for physical/chemical plausibility."""
import torch
import numpy as np
from typing import Dict, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

class StructureValidator:
    """Validate protein structures for physical plausibility."""
    
    def __init__(self):
        # Standard bond lengths (Angstroms)
        self.bond_lengths = {
            'CA-CA': (3.8, 3.8),  # Alpha carbon distance
            'CA-N': (1.46, 1.47),
            'CA-C': (1.52, 1.53),
            'C-N': (1.32, 1.33),  # Peptide bond
        }
        
        # Standard bond angles (degrees)
        self.bond_angles = {
            'N-CA-C': (110, 115),
            'CA-C-N': (115, 120),
            'C-N-CA': (120, 125),
        }
        
        # Ramachandran plot allowed regions (phi, psi)
        self.ramachandran_regions = {
            'alpha_helix': {'phi': (-70, -50), 'psi': (-50, -30)},
            'beta_sheet': {'phi': (-140, -110), 'psi': (120, 150)},
            'left_alpha': {'phi': (50, 70), 'psi': (30, 50)},
        }
    
    def check_bond_lengths(self, coords: torch.Tensor) -> Dict:
        """Verify CA-CA distances are reasonable."""
        ca_dists = torch.sqrt(torch.sum(
            (coords[1:] - coords[:-1]) ** 2, dim=-1
        ))
        
        expected_min, expected_max = self.bond_lengths['CA-CA']
        violations = ((ca_dists < expected_min - 0.5) | 
                     (ca_dists > expected_max + 0.5)).sum().item()
        
        return {
            'mean_ca_distance': ca_dists.mean().item(),
            'std_ca_distance': ca_dists.std().item(),
            'violations': violations,
            'violation_rate': violations / len(ca_dists) if len(ca_dists) > 0 else 0,
            'valid': violations < len(ca_dists) * 0.1  # Less than 10% violations
        }
    
    def check_bond_angles(self, coords: torch.Tensor) -> Dict:
        """Check if bond angles are in reasonable ranges."""
        if len(coords) < 3:
            return {'valid': True, 'mean_angle': 0, 'violations': 0}
        
        # Vectors between consecutive CA atoms
        v1 = coords[1:-1] - coords[:-2]
        v2 = coords[2:] - coords[1:-1]
        
        # Angles using dot product
        cos_angles = torch.sum(v1 * v2, dim=-1) / (
            torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + 1e-8
        )
        cos_angles = torch.clamp(cos_angles, -1, 1)
        angles = torch.acos(cos_angles) * 180 / np.pi
        
        # Expected range for CA-CA-CA angles
        violations = ((angles < 80) | (angles > 140)).sum().item()
        
        return {
            'mean_angle': angles.mean().item(),
            'std_angle': angles.std().item(),
            'violations': violations,
            'violation_rate': violations / len(angles) if len(angles) > 0 else 0,
            'valid': violations < len(angles) * 0.15
        }
    
    def check_clash_free(self, coords: torch.Tensor, threshold: float = 2.0) -> Dict:
        """Check for atomic clashes (atoms too close)."""
        # Pairwise distances
        dists = torch.cdist(coords, coords)
        
        # Mask out adjacent residues (they're supposed to be close)
        mask = torch.ones_like(dists, dtype=torch.bool)
        for i in range(len(coords)):
            for j in range(max(0, i-2), min(len(coords), i+3)):
                mask[i, j] = False
        
        # Count clashes
        clashes = ((dists < threshold) & mask).sum().item() // 2
        
        return {
            'n_clashes': clashes,
            'clash_rate': clashes / (len(coords) * (len(coords) - 3) / 2) if len(coords) > 3 else 0,
            'valid': clashes == 0
        }
    
    def check_compactness(self, coords: torch.Tensor) -> Dict:
        """Verify structure is reasonably compact (not exploded)."""
        center = coords.mean(dim=0)
        dists_from_center = torch.sqrt(torch.sum((coords - center) ** 2, dim=-1))
        radius_of_gyration = torch.sqrt(torch.mean(dists_from_center ** 2)).item()
        
        # Expected Rg for a globular protein: ~2.2 * N^0.38 Angstroms
        expected_rg = 2.2 * (len(coords) ** 0.38)
        
        return {
            'radius_of_gyration': radius_of_gyration,
            'expected_rg': expected_rg,
            'rg_ratio': radius_of_gyration / expected_rg,
            'valid': 0.5 < radius_of_gyration / expected_rg < 2.0
        }
    
    def check_secondary_structure_consistency(self, coords: torch.Tensor) -> Dict:
        """Check if predicted structure has reasonable secondary structure."""
        if len(coords) < 4:
            return {'valid': True, 'helix_content': 0, 'sheet_content': 0}
        
        # Simple helix detection: CA(i) to CA(i+3) distance ~5.5 Å
        helix_dists = torch.sqrt(torch.sum(
            (coords[3:] - coords[:-3]) ** 2, dim=-1
        ))
        helix_like = ((helix_dists > 4.5) & (helix_dists < 6.5)).float().mean().item()
        
        # Simple sheet detection: extended conformation, CA-CA ~3.8 Å
        ca_dists = torch.sqrt(torch.sum(
            (coords[1:] - coords[:-1]) ** 2, dim=-1
        ))
        extended = ((ca_dists > 3.5) & (ca_dists < 4.2)).float().mean().item()
        
        return {
            'helix_content': helix_like,
            'extended_content': extended,
            'structured_content': helix_like + extended,
            'valid': (helix_like + extended) > 0.3  # At least 30% structured
        }
    
    def validate_structure(self, coords: torch.Tensor) -> Dict:
        """Run all validation checks."""
        results = {
            'bond_lengths': self.check_bond_lengths(coords),
            'bond_angles': self.check_bond_angles(coords),
            'clash_free': self.check_clash_free(coords),
            'compactness': self.check_compactness(coords),
            'secondary_structure': self.check_secondary_structure_consistency(coords)
        }
        
        # Overall validity
        results['overall_valid'] = all(
            check['valid'] for check in results.values()
        )
        
        results['validity_score'] = sum(
            1.0 if check['valid'] else 0.0 
            for check in results.values()
        ) / len(results)
        
        return results
    
    def print_report(self, validation_results: Dict):
        """Print human-readable validation report."""
        print("\n🔬 Structure Validation Report")
        print("=" * 60)
        
        print("\n📏 Bond Lengths:")
        bl = validation_results['bond_lengths']
        print(f"  Mean CA-CA distance: {bl['mean_ca_distance']:.2f} Å")
        print(f"  Violations: {bl['violations']} ({bl['violation_rate']*100:.1f}%)")
        print(f"  Status: {'✅ PASS' if bl['valid'] else '❌ FAIL'}")
        
        print("\n📐 Bond Angles:")
        ba = validation_results['bond_angles']
        print(f"  Mean angle: {ba['mean_angle']:.1f}°")
        print(f"  Violations: {ba['violations']} ({ba['violation_rate']*100:.1f}%)")
        print(f"  Status: {'✅ PASS' if ba['valid'] else '❌ FAIL'}")
        
        print("\n💥 Clash Check:")
        cf = validation_results['clash_free']
        print(f"  Clashes detected: {cf['n_clashes']}")
        print(f"  Status: {'✅ PASS' if cf['valid'] else '❌ FAIL'}")
        
        print("\n🎯 Compactness:")
        comp = validation_results['compactness']
        print(f"  Radius of gyration: {comp['radius_of_gyration']:.2f} Å")
        print(f"  Expected: {comp['expected_rg']:.2f} Å")
        print(f"  Ratio: {comp['rg_ratio']:.2f}")
        print(f"  Status: {'✅ PASS' if comp['valid'] else '❌ FAIL'}")
        
        print("\n🧬 Secondary Structure:")
        ss = validation_results['secondary_structure']
        print(f"  Helix-like content: {ss['helix_content']*100:.1f}%")
        print(f"  Extended content: {ss['extended_content']*100:.1f}%")
        print(f"  Status: {'✅ PASS' if ss['valid'] else '❌ FAIL'}")
        
        print("\n" + "=" * 60)
        print(f"Overall Validity Score: {validation_results['validity_score']*100:.1f}%")
        print(f"Status: {'✅ VALID STRUCTURE' if validation_results['overall_valid'] else '❌ INVALID STRUCTURE'}")
        print("=" * 60 + "\n")


if __name__ == '__main__':
    # Test with synthetic structure
    coords = torch.randn(50, 3) * 10  # Random structure
    
    validator = StructureValidator()
    results = validator.validate_structure(coords)
    validator.print_report(results)
