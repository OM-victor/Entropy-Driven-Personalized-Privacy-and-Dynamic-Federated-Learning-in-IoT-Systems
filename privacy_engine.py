import torch
import numpy as np
from typing import List, Dict, Any
from options import parse_args
from utils import compute_noise_multiplier

args = parse_args()

class DifferentialPrivacyEngine:
    def __init__(self, target_epsilon: float, target_delta: float, clipping_bound: float):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.clipping_bound = clipping_bound
        self.noise_multiplier = None
    
    def compute_noise_multiplier(self, global_epoch: int, local_epoch: int, batch_size: int, client_data_sizes: List[int]) -> float:
        self.noise_multiplier = compute_noise_multiplier(
            self.target_epsilon,
            self.target_delta,
            global_epoch,
            local_epoch,
            batch_size,
            client_data_sizes
        )
        return self.noise_multiplier
    
    def clip_gradients(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        if args.no_clip:
            return gradients
        
        total_norm = torch.sqrt(sum([torch.sum(grad ** 2) for grad in gradients]))
        clip_coef = self.clipping_bound / (total_norm + 1e-6)
        clip_coef = min(1.0, clip_coef)
        
        clipped_gradients = [grad * clip_coef for grad in gradients]
        return clipped_gradients
    
    def add_noise(self, gradients: List[torch.Tensor], num_clients: int) -> List[torch.Tensor]:
        if args.no_noise or self.noise_multiplier is None:
            return gradients
        
        noise_stddev = self.clipping_bound * self.noise_multiplier / np.sqrt(num_clients)
        noisy_gradients = []
        
        for grad in gradients:
            noise = torch.randn_like(grad) * noise_stddev
            noisy_gradients.append(grad + noise)
        
        return noisy_gradients
    
    def apply_differential_privacy(self, gradients: List[torch.Tensor], num_clients: int) -> List[torch.Tensor]:
        clipped_gradients = self.clip_gradients(gradients)
        private_gradients = self.add_noise(clipped_gradients, num_clients)
        return private_gradients
    
    def get_privacy_budget(self) -> Dict[str, float]:
        return {
            'epsilon': self.target_epsilon,
            'delta': self.target_delta,
            'noise_multiplier': self.noise_multiplier
        }

class AdaptiveClipping:
    def __init__(self, initial_bound: float, target_quantile: float = 0.9):
        self.clipping_bound = initial_bound
        self.target_quantile = target_quantile
        self.gradient_norms = []
    
    def update_clipping_bound(self, gradients: List[torch.Tensor]):
        if not gradients:
            return
        
        total_norm = torch.sqrt(sum([torch.sum(grad ** 2) for grad in gradients]))
        self.gradient_norms.append(total_norm.item())
        
        if len(self.gradient_norms) >= 10:
            quantile_value = np.quantile(self.gradient_norms, self.target_quantile)
            self.clipping_bound = quantile_value
            self.gradient_norms = self.gradient_norms[-5:]
    
    def get_clipping_bound(self) -> float:
        return self.clipping_bound

class PrivacyAccountant:
    def __init__(self):
        self.epsilon_used = 0.0
        self.delta_used = 0.0
        self.steps_taken = 0
    
    def update_privacy_budget(self, epsilon_step: float, delta_step: float):
        self.epsilon_used += epsilon_step
        self.delta_used += delta_step
        self.steps_taken += 1
    
    def get_remaining_budget(self, target_epsilon: float, target_delta: float) -> Dict[str, float]:
        return {
            'remaining_epsilon': max(0, target_epsilon - self.epsilon_used),
            'remaining_delta': max(0, target_delta - self.delta_used),
            'epsilon_used': self.epsilon_used,
            'delta_used': self.delta_used,
            'steps_taken': self.steps_taken
        }
    
    def is_budget_exhausted(self, target_epsilon: float, target_delta: float) -> bool:
        return self.epsilon_used >= target_epsilon or self.delta_used >= target_delta

class GaussianMechanism:
    def __init__(self, sensitivity: float, epsilon: float, delta: float):
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta
        self.sigma = self._compute_sigma()
    
    def _compute_sigma(self) -> float:
        return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def add_noise(self, data: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(data) * self.sigma
        return data + noise

class LaplaceMechanism:
    def __init__(self, sensitivity: float, epsilon: float):
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.scale = self.sensitivity / self.epsilon
    
    def add_noise(self, data: torch.Tensor) -> torch.Tensor:
        noise = torch.distributions.Laplace(0, self.scale).sample(data.shape)
        return data + noise
