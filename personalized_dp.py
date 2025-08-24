import torch
import numpy as np
from typing import List, Dict, Any
from options import parse_args
from entropy_calculator import EntropyCalculator

args = parse_args()

class PersonalizedDifferentialPrivacy:
    def __init__(self, total_epsilon: float, delta: float, clipping_bound: float):
        self.total_epsilon = total_epsilon
        self.delta = delta
        self.clipping_bound = clipping_bound
        self.entropy_calculator = EntropyCalculator()
        self.client_entropies = []
        self.client_epsilons = []
        self.client_noise_scales = []
    
    def allocate_privacy_budgets(self, clients_datasets: List, num_classes: int) -> List[float]:
        self.client_entropies = self.entropy_calculator.calculate_all_clients_entropy(
            clients_datasets, num_classes
        )
        
        normalized_entropies = self.entropy_calculator.normalize_entropies(self.client_entropies)
        
        self.client_epsilons = []
        for normalized_entropy in normalized_entropies:
            epsilon_i = self.total_epsilon * normalized_entropy
            self.client_epsilons.append(epsilon_i)
        
        self._calculate_noise_scales()
        
        return self.client_epsilons
    
    def _calculate_noise_scales(self):
        self.client_noise_scales = []
        for epsilon_i in self.client_epsilons:
            if epsilon_i > 0:
                sigma_i = 2.0 * np.sqrt(2 * np.log(1.25 / self.delta)) / epsilon_i
                sigma_i = 9.43 / epsilon_i
            else:
                sigma_i = float('inf')
            self.client_noise_scales.append(sigma_i)
    
    def apply_personalized_noise(self, client_id: int, model_update: List[torch.Tensor]) -> List[torch.Tensor]:
        if client_id >= len(self.client_noise_scales):
            return model_update
        
        noise_scale = self.client_noise_scales[client_id]
        if noise_scale == float('inf'):
            return model_update
        
        noisy_update = []
        for param in model_update:
            noise = torch.randn_like(param) * noise_scale
            noisy_update.append(param + noise)
        
        return noisy_update
    
    def clip_gradients(self, model_update: List[torch.Tensor]) -> List[torch.Tensor]:
        if args.no_clip:
            return model_update
        
        total_norm = torch.sqrt(sum([torch.sum(param ** 2) for param in model_update]))
        clip_coef = self.clipping_bound / (total_norm + 1e-6)
        clip_coef = min(1.0, clip_coef)
        
        clipped_update = [param * clip_coef for param in model_update]
        return clipped_update
    
    def apply_personalized_dp(self, client_id: int, model_update: List[torch.Tensor]) -> List[torch.Tensor]:
        clipped_update = self.clip_gradients(model_update)
        private_update = self.apply_personalized_noise(client_id, clipped_update)
        return private_update
    
    def get_privacy_info(self, client_id: int) -> Dict[str, float]:
        if client_id >= len(self.client_epsilons):
            return {}
        
        return {
            'entropy': self.client_entropies[client_id],
            'epsilon': self.client_epsilons[client_id],
            'noise_scale': self.client_noise_scales[client_id]
        }
    
    def get_all_privacy_info(self) -> Dict[str, Any]:
        return {
            'entropies': self.client_entropies,
            'epsilons': self.client_epsilons,
            'noise_scales': self.client_noise_scales,
            'statistics': self.entropy_calculator.get_entropy_statistics(self.client_entropies)
        }

class AdaptivePersonalizedDP(PersonalizedDifferentialPrivacy):
    def __init__(self, total_epsilon: float, delta: float, clipping_bound: float, adaptation_rate: float = 0.1):
        super().__init__(total_epsilon, delta, clipping_bound)
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
    
    def update_privacy_budgets(self, performance_metrics: List[float]):
        self.performance_history.append(performance_metrics)
        
        if len(self.performance_history) < 3:
            return
        
        recent_performance = np.mean(self.performance_history[-3:], axis=0)
        
        for i in range(len(self.client_epsilons)):
            if recent_performance[i] < np.mean(recent_performance):
                self.client_epsilons[i] *= (1 + self.adaptation_rate)
            else:
                self.client_epsilons[i] *= (1 - self.adaptation_rate * 0.5)
        
        self._calculate_noise_scales()
