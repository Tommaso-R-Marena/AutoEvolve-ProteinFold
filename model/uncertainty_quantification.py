"""Uncertainty quantification for protein structure predictions.

Provides multiple methods for estimating prediction confidence:
1. Ensemble predictions
2. Monte Carlo dropout
3. Evidential deep learning
4. Conformal prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np

class EnsemblePredictor:
    """Make predictions using an ensemble of models."""
    
    def __init__(self, models: List[nn.Module]):
        self.models = models
        self.num_models = len(models)
    
    @torch.no_grad()
    def predict(self, sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate ensemble predictions with uncertainty estimates.
        
        Returns:
            mean_coords: [B, L, 3] - mean prediction
            std_coords: [B, L, 3] - standard deviation per dimension
            mean_confidence: [B, L] - mean confidence
            ensemble_uncertainty: [B, L] - prediction variance
        """
        all_coords = []
        all_confidences = []
        
        for model in self.models:
            model.eval()
            outputs = model(sequence)
            all_coords.append(outputs['coordinates'])
            all_confidences.append(outputs['confidence'])
        
        # Stack predictions
        coords_stack = torch.stack(all_coords, dim=0)  # [N, B, L, 3]
        conf_stack = torch.stack(all_confidences, dim=0)  # [N, B, L]
        
        # Compute statistics
        mean_coords = coords_stack.mean(dim=0)
        std_coords = coords_stack.std(dim=0)
        mean_confidence = conf_stack.mean(dim=0)
        
        # Ensemble uncertainty (coordinate variance)
        ensemble_uncertainty = torch.sum(std_coords ** 2, dim=-1).sqrt()
        
        return {
            'coordinates': mean_coords,
            'coordinate_std': std_coords,
            'confidence': mean_confidence,
            'uncertainty': ensemble_uncertainty,
            'all_predictions': coords_stack
        }

class MCDropoutPredictor:
    """Monte Carlo Dropout for uncertainty estimation."""
    
    def __init__(self, model: nn.Module, n_samples: int = 20):
        self.model = model
        self.n_samples = n_samples
        self._enable_dropout()
    
    def _enable_dropout(self):
        """Enable dropout during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    @torch.no_grad()
    def predict(self, sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate predictions with MC dropout."""
        self._enable_dropout()
        
        all_coords = []
        all_confidences = []
        
        for _ in range(self.n_samples):
            outputs = self.model(sequence)
            all_coords.append(outputs['coordinates'])
            all_confidences.append(outputs['confidence'])
        
        coords_stack = torch.stack(all_coords, dim=0)
        conf_stack = torch.stack(all_confidences, dim=0)
        
        mean_coords = coords_stack.mean(dim=0)
        std_coords = coords_stack.std(dim=0)
        mean_confidence = conf_stack.mean(dim=0)
        epistemic_uncertainty = torch.sum(std_coords ** 2, dim=-1).sqrt()
        
        return {
            'coordinates': mean_coords,
            'coordinate_std': std_coords,
            'confidence': mean_confidence,
            'epistemic_uncertainty': epistemic_uncertainty
        }

class EvidentialNetwork(nn.Module):
    """Evidential deep learning for uncertainty quantification.
    
    Learns to predict parameters of a Normal-Inverse-Gamma distribution,
    which provides both aleatoric and epistemic uncertainty.
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        # Predict 4 evidential parameters per coordinate
        self.coord_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 12)  # 4 params × 3 dimensions (x,y,z)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict evidential parameters.
        
        Returns:
            gamma: location
            nu: degrees of freedom (epistemic uncertainty)
            alpha: shape
            beta: scale (aleatoric uncertainty)
        """
        params = self.coord_net(x)  # [B, L, 12]
        
        # Split into 4 parameters for each of 3 dimensions
        params = params.view(*params.shape[:-1], 3, 4)  # [B, L, 3, 4]
        
        gamma = params[..., 0]  # Mean prediction [B, L, 3]
        nu = F.softplus(params[..., 1]) + 1  # > 0
        alpha = F.softplus(params[..., 2]) + 1  # > 0  
        beta = F.softplus(params[..., 3])  # > 0
        
        # Aleatoric uncertainty (data noise)
        aleatoric = beta / (alpha - 1)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = beta / (nu * (alpha - 1))
        
        return {
            'coordinates': gamma,
            'aleatoric_uncertainty': aleatoric,
            'epistemic_uncertainty': epistemic,
            'total_uncertainty': aleatoric + epistemic
        }
    
    def nig_loss(self, predictions: Dict, targets: torch.Tensor) -> torch.Tensor:
        """Normal-Inverse-Gamma negative log-likelihood loss."""
        gamma = predictions['coordinates']
        nu = predictions['nu']
        alpha = predictions['alpha']
        beta = predictions['beta']
        
        # NIG-NLL loss
        error = (targets - gamma) ** 2
        loss = 0.5 * torch.log(torch.pi / nu) \
             - alpha * torch.log(2 * beta) \
             + (alpha + 0.5) * torch.log(nu * error + 2 * beta) \
             + torch.lgamma(alpha) \
             - torch.lgamma(alpha + 0.5)
        
        return loss.mean()

class ConformalPredictor:
    """Conformal prediction for calibrated uncertainty intervals.
    
    Provides statistically valid prediction intervals with guaranteed coverage.
    """
    
    def __init__(self, model: nn.Module, calibration_data: List[Tuple]):
        self.model = model
        self.nonconformity_scores = self._compute_scores(calibration_data)
    
    def _compute_scores(self, calibration_data: List[Tuple]) -> np.ndarray:
        """Compute nonconformity scores on calibration set."""
        scores = []
        
        with torch.no_grad():
            for sequence, true_coords in calibration_data:
                pred = self.model(sequence.unsqueeze(0))['coordinates']
                error = torch.norm(pred[0] - true_coords, dim=-1)
                scores.extend(error.cpu().numpy())
        
        return np.array(scores)
    
    def predict_with_interval(self, sequence: torch.Tensor, alpha: float = 0.1) -> Dict:
        """Predict with conformal prediction interval.
        
        Args:
            sequence: input sequence
            alpha: significance level (e.g., 0.1 for 90% coverage)
        
        Returns:
            prediction with calibrated uncertainty bounds
        """
        with torch.no_grad():
            pred = self.model(sequence)
        
        # Compute quantile of nonconformity scores
        n = len(self.nonconformity_scores)
        quantile_idx = int(np.ceil((n + 1) * (1 - alpha)))
        threshold = np.sort(self.nonconformity_scores)[min(quantile_idx, n - 1)]
        
        return {
            'coordinates': pred['coordinates'],
            'confidence': pred['confidence'],
            'prediction_radius': threshold,  # Guaranteed coverage radius
            'coverage_probability': 1 - alpha
        }

class UncertaintyAwareLoss(nn.Module):
    """Loss function that accounts for prediction uncertainty."""
    
    def __init__(self, coord_weight: float = 1.0, uncertainty_weight: float = 0.1):
        super().__init__()
        self.coord_weight = coord_weight
        self.uncertainty_weight = uncertainty_weight
    
    def forward(self, predictions: Dict, targets: Dict, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute uncertainty-aware loss."""
        # Coordinate loss
        coord_diff = (predictions['coordinates'] - targets['coordinates']) ** 2
        coord_diff = coord_diff * mask.unsqueeze(-1)
        coord_loss = coord_diff.sum() / mask.sum()
        
        # Uncertainty regularization (prevent overconfidence)
        if 'uncertainty' in predictions:
            uncertainty = predictions['uncertainty']
            # Penalize very low uncertainty (overconfidence)
            uncertainty_reg = -torch.log(uncertainty + 1e-8).mean()
        else:
            uncertainty_reg = torch.tensor(0.0, device=coord_loss.device)
        
        # Calibration loss (if epistemic uncertainty available)
        if 'epistemic_uncertainty' in predictions:
            epistemic = predictions['epistemic_uncertainty']
            # Epistemic uncertainty should correlate with error
            errors = torch.norm(coord_diff, dim=-1).sqrt()
            calibration_loss = F.mse_loss(epistemic, errors)
        else:
            calibration_loss = torch.tensor(0.0, device=coord_loss.device)
        
        total_loss = (
            self.coord_weight * coord_loss +
            self.uncertainty_weight * uncertainty_reg +
            0.1 * calibration_loss
        )
        
        return total_loss, {
            'coord_loss': coord_loss.item(),
            'uncertainty_reg': uncertainty_reg.item() if torch.is_tensor(uncertainty_reg) else 0.0,
            'calibration_loss': calibration_loss.item() if torch.is_tensor(calibration_loss) else 0.0
        }
