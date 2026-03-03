#!/usr/bin/env python3
"""Comprehensive benchmarking suite for protein structure prediction.

Tests model on multiple metrics:
- RMSD (Root Mean Square Deviation)
- TM-score (Template Modeling score)
- GDT (Global Distance Test)
- lDDT (local Distance Difference Test)
- CAMEO-style assessment
- Speed benchmarks
"""
import torch
import numpy as np
from typing import Dict, List, Tuple
import time
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))

class StructureMetrics:
    """Compute comprehensive structure quality metrics."""
    
    @staticmethod
    def rmsd(pred: np.ndarray, true: np.ndarray, mask: np.ndarray = None) -> float:
        """Compute CA RMSD after optimal alignment."""
        if mask is not None:
            pred = pred[mask]
            true = true[mask]
        
        # Center structures
        pred_centered = pred - pred.mean(axis=0)
        true_centered = true - true.mean(axis=0)
        
        # Kabsch algorithm for optimal rotation
        correlation = pred_centered.T @ true_centered
        u, s, vt = np.linalg.svd(correlation)
        rotation = vt.T @ u.T
        
        # Apply rotation
        pred_aligned = pred_centered @ rotation
        
        # Compute RMSD
        rmsd = np.sqrt(np.mean(np.sum((pred_aligned - true_centered) ** 2, axis=1)))
        return float(rmsd)
    
    @staticmethod
    def tm_score(pred: np.ndarray, true: np.ndarray, mask: np.ndarray = None) -> float:
        """Compute TM-score (Template Modeling score).
        
        TM-score ranges from 0 to 1, where:
        - TM > 0.5 indicates similar fold
        - TM > 0.6 indicates high confidence match
        """
        if mask is not None:
            pred = pred[mask]
            true = true[mask]
        
        L = len(pred)
        d0 = 1.24 * (L - 15) ** (1/3) - 1.8  # Normalization factor
        
        # Compute distances after alignment
        distances = np.sqrt(np.sum((pred - true) ** 2, axis=1))
        
        # TM-score formula
        tm = np.sum(1 / (1 + (distances / d0) ** 2)) / L
        
        return float(tm)
    
    @staticmethod
    def gdt_ts(pred: np.ndarray, true: np.ndarray, mask: np.ndarray = None) -> Dict[str, float]:
        """Compute GDT (Global Distance Test) scores.
        
        Returns GDT_TS (Total Score) and GDT_HA (High Accuracy).
        """
        if mask is not None:
            pred = pred[mask]
            true = true[mask]
        
        distances = np.sqrt(np.sum((pred - true) ** 2, axis=1))
        L = len(pred)
        
        # GDT_TS: average of fractions within 1, 2, 4, 8Å
        thresholds_ts = [1.0, 2.0, 4.0, 8.0]
        gdt_ts = np.mean([np.sum(distances < t) / L for t in thresholds_ts]) * 100
        
        # GDT_HA: average of fractions within 0.5, 1, 2, 4Å
        thresholds_ha = [0.5, 1.0, 2.0, 4.0]
        gdt_ha = np.mean([np.sum(distances < t) / L for t in thresholds_ha]) * 100
        
        return {
            'GDT_TS': float(gdt_ts),
            'GDT_HA': float(gdt_ha)
        }
    
    @staticmethod
    def lddt(pred: np.ndarray, true: np.ndarray, mask: np.ndarray = None, 
            radius: float = 15.0) -> float:
        """Compute lDDT (local Distance Difference Test).
        
        lDDT ranges from 0 to 100, measuring local geometry preservation.
        """
        if mask is not None:
            pred = pred[mask]
            true = true[mask]
        
        L = len(pred)
        
        # Compute pairwise distances
        pred_dist = np.sqrt(np.sum((pred[:, None] - pred[None, :]) ** 2, axis=-1))
        true_dist = np.sqrt(np.sum((true[:, None] - true[None, :]) ** 2, axis=-1))
        
        # Consider only distances within radius
        mask_radius = true_dist < radius
        
        # Distance differences
        diff = np.abs(pred_dist - true_dist)
        
        # Count preserved contacts at different thresholds
        thresholds = [0.5, 1.0, 2.0, 4.0]
        preserved = np.mean([np.sum((diff < t) & mask_radius) / np.sum(mask_radius) 
                            for t in thresholds])
        
        return float(preserved * 100)

class BenchmarkSuite:
    """Comprehensive benchmark for protein folding models."""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.metrics_calculator = StructureMetrics()
        self.results = []
    
    def evaluate_protein(self, sequence: torch.Tensor, true_coords: np.ndarray, 
                        protein_id: str = "") -> Dict:
        """Evaluate model on a single protein."""
        self.model.eval()
        
        # Timing
        start_time = time.time()
        
        with torch.no_grad():
            predictions = self.model(sequence.to(self.device))
        
        inference_time = time.time() - start_time
        
        # Convert to numpy
        pred_coords = predictions['coordinates'][0].cpu().numpy()
        confidence = predictions['confidence'][0].cpu().numpy()
        
        # Compute all metrics
        rmsd = self.metrics_calculator.rmsd(pred_coords, true_coords)
        tm_score = self.metrics_calculator.tm_score(pred_coords, true_coords)
        gdt = self.metrics_calculator.gdt_ts(pred_coords, true_coords)
        lddt = self.metrics_calculator.lddt(pred_coords, true_coords)
        
        result = {
            'protein_id': protein_id,
            'length': len(true_coords),
            'rmsd': rmsd,
            'tm_score': tm_score,
            'gdt_ts': gdt['GDT_TS'],
            'gdt_ha': gdt['GDT_HA'],
            'lddt': lddt,
            'mean_confidence': float(confidence.mean()),
            'inference_time': inference_time,
            'inference_speed': len(true_coords) / inference_time  # residues/sec
        }
        
        self.results.append(result)
        return result
    
    def run_benchmark_suite(self, test_set: List[Tuple[torch.Tensor, np.ndarray, str]]) -> Dict:
        """Run full benchmark on test set.
        
        Args:
            test_set: List of (sequence, true_coords, protein_id)
        
        Returns:
            Aggregated statistics
        """
        print(f"\n{'='*60}")
        print("🧪 Running Comprehensive Benchmark")
        print(f"{'='*60}")
        print(f"Test set size: {len(test_set)} proteins\n")
        
        for i, (sequence, true_coords, protein_id) in enumerate(test_set):
            result = self.evaluate_protein(sequence, true_coords, protein_id)
            
            print(f"[{i+1}/{len(test_set)}] {protein_id} (L={result['length']})")
            print(f"  RMSD: {result['rmsd']:.2f}Å | TM: {result['tm_score']:.3f} | "
                  f"GDT_TS: {result['gdt_ts']:.1f} | lDDT: {result['lddt']:.1f}")
            print(f"  Time: {result['inference_time']:.2f}s | "
                  f"Speed: {result['inference_speed']:.0f} res/s\n")
        
        # Aggregate statistics
        stats = self._compute_statistics()
        self._print_summary(stats)
        
        return stats
    
    def _compute_statistics(self) -> Dict:
        """Compute aggregate statistics."""
        if not self.results:
            return {}
        
        metrics = ['rmsd', 'tm_score', 'gdt_ts', 'gdt_ha', 'lddt', 'mean_confidence', 'inference_time']
        stats = {}
        
        for metric in metrics:
            values = [r[metric] for r in self.results]
            stats[metric] = {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        # Quality categories (based on TM-score)
        tm_scores = [r['tm_score'] for r in self.results]
        stats['quality_distribution'] = {
            'high_quality (TM>0.6)': sum(tm > 0.6 for tm in tm_scores),
            'medium_quality (0.4<TM<0.6)': sum(0.4 < tm <= 0.6 for tm in tm_scores),
            'low_quality (TM<0.4)': sum(tm <= 0.4 for tm in tm_scores)
        }
        
        return stats
    
    def _print_summary(self, stats: Dict):
        """Print benchmark summary."""
        print(f"\n{'='*60}")
        print("📊 Benchmark Summary")
        print(f"{'='*60}\n")
        
        print(f"RMSD:      {stats['rmsd']['mean']:.2f} ± {stats['rmsd']['std']:.2f}Å")
        print(f"TM-score:  {stats['tm_score']['mean']:.3f} ± {stats['tm_score']['std']:.3f}")
        print(f"GDT_TS:    {stats['gdt_ts']['mean']:.1f} ± {stats['gdt_ts']['std']:.1f}")
        print(f"GDT_HA:    {stats['gdt_ha']['mean']:.1f} ± {stats['gdt_ha']['std']:.1f}")
        print(f"lDDT:      {stats['lddt']['mean']:.1f} ± {stats['lddt']['std']:.1f}")
        print(f"Confidence: {stats['mean_confidence']['mean']:.3f}")
        print(f"Speed:     {1/stats['inference_time']['mean']:.1f} proteins/sec\n")
        
        print("Quality Distribution:")
        for category, count in stats['quality_distribution'].items():
            print(f"  {category}: {count}")
        
        print(f"\n{'='*60}\n")
    
    def save_results(self, output_path: str):
        """Save detailed results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        stats = self._compute_statistics()
        
        output = {
            'individual_results': self.results,
            'aggregate_statistics': stats,
            'metadata': {
                'num_proteins': len(self.results),
                'model_generation': getattr(self.model, 'generation', 0)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✅ Results saved to {output_path}")

if __name__ == '__main__':
    print("Benchmark suite loaded. Import and use in your evaluation scripts.")
