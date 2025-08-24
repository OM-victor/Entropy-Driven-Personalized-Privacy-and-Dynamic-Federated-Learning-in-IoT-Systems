import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from options import parse_args
import json
import os

args = parse_args()

class DynamicWeightCompression:
    def __init__(self, c_max: float = 0.9999, c_min: float = 0.99, sigma: float = 10.0, 
                 w_a: float = 0.5, w_b: float = 0.5, residual_threshold: float = 0.1):
        self.c_max = c_max
        self.c_min = c_min
        self.sigma = sigma
        self.w_a = w_a
        self.w_b = w_b
        self.residual_threshold = residual_threshold
        self.client_performance_db = {}
        
    def load_previous_performance(self, client_id: int) -> Tuple[float, float]:
        if client_id not in self.client_performance_db:
            return 0.0, float('inf')
        
        db_entry = self.client_performance_db[client_id]
        return db_entry.get('accuracy', 0.0), db_entry.get('loss', float('inf'))
    
    def save_current_performance(self, client_id: int, accuracy: float, loss: float):
        self.client_performance_db[client_id] = {
            'accuracy': accuracy,
            'loss': loss
        }
    
    def calculate_performance_change(self, accuracy_t_minus_1: float, loss_t_minus_1: float,
                                   accuracy_t: float, loss_t: float) -> float:
        delta_accuracy = abs(accuracy_t_minus_1 - accuracy_t) / 100.0
        delta_loss = abs(loss_t_minus_1 - loss_t)
        
        delta_p = self.w_a * delta_accuracy + self.w_b * delta_loss
        return delta_p
    
    def calculate_compression_ratio(self, delta_p: float) -> float:
        compression_ratio = self.c_max - (2 * (self.c_max - self.c_min)) / (1 + np.exp(delta_p * self.sigma))
        return max(self.c_min, min(self.c_max, compression_ratio))
    
    def calculate_compression_sensitivity(self, delta_p: float) -> float:
        exp_term = np.exp(delta_p * self.sigma)
        sensitivity = 2 * (self.c_max - self.c_min) * self.sigma * exp_term / ((1 + exp_term) ** 2)
        return sensitivity
    
    def top_k_compression(self, model_update: List[torch.Tensor], compression_ratio: float) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        total_params = sum(param.numel() for param in model_update)
        k = int(compression_ratio * total_params)
        
        all_params = []
        param_shapes = []
        param_indices = []
        
        for i, param in enumerate(model_update):
            param_flat = param.flatten()
            all_params.append(param_flat)
            param_shapes.append(param.shape)
            param_indices.extend([i] * param_flat.numel())
        
        all_params_flat = torch.cat(all_params)
        abs_values = torch.abs(all_params_flat)
        
        _, top_indices = torch.topk(abs_values, k)
        
        compressed_update = []
        residual_update = []
        
        for i, param in enumerate(model_update):
            compressed_param = torch.zeros_like(param)
            residual_param = param.clone()
            
            param_flat = param.flatten()
            start_idx = sum(p.numel() for p in model_update[:i])
            end_idx = start_idx + param.numel()
            
            for top_idx in top_indices:
                if start_idx <= top_idx < end_idx:
                    local_idx = top_idx - start_idx
                    compressed_param.view(-1)[local_idx] = param_flat[local_idx]
                    residual_param.view(-1)[local_idx] = 0.0
            
            compressed_update.append(compressed_param)
            residual_update.append(residual_param)
        
        return compressed_update, residual_update
    
    def apply_dynamic_compression(self, client_id: int, model_update: List[torch.Tensor], 
                                current_accuracy: float, current_loss: float) -> Tuple[List[torch.Tensor], List[torch.Tensor], float]:
        
        accuracy_t_minus_1, loss_t_minus_1 = self.load_previous_performance(client_id)
        
        delta_p = self.calculate_performance_change(accuracy_t_minus_1, loss_t_minus_1, 
                                                  current_accuracy, current_loss)
        
        compression_ratio = self.calculate_compression_ratio(delta_p)
        sensitivity = self.calculate_compression_sensitivity(delta_p)
        
        compressed_update, residual_update = self.top_k_compression(model_update, compression_ratio)
        
        self.save_current_performance(client_id, current_accuracy, current_loss)
        
        return compressed_update, residual_update, compression_ratio
    
    def accumulate_residual(self, client_id: int, residual_update: List[torch.Tensor]) -> List[torch.Tensor]:
        residual_key = f"residual_{client_id}"
        
        if residual_key not in self.client_performance_db:
            self.client_performance_db[residual_key] = [torch.zeros_like(residual) for residual in residual_update]
        
        accumulated_residual = self.client_performance_db[residual_key]
        
        for i, (acc_res, new_res) in enumerate(zip(accumulated_residual, residual_update)):
            acc_res.add_(new_res)
        
        return accumulated_residual
    
    def check_residual_threshold(self, accumulated_residual: List[torch.Tensor]) -> bool:
        total_residual_norm = sum(torch.norm(residual) for residual in accumulated_residual)
        return total_residual_norm > self.residual_threshold
    
    def clear_residual(self, client_id: int):
        residual_key = f"residual_{client_id}"
        if residual_key in self.client_performance_db:
            del self.client_performance_db[residual_key]
    
    def get_compression_stats(self, client_id: int) -> Dict[str, Any]:
        if client_id not in self.client_performance_db:
            return {}
        
        db_entry = self.client_performance_db[client_id]
        return {
            'accuracy': db_entry.get('accuracy', 0.0),
            'loss': db_entry.get('loss', 0.0),
            'compression_ratio': db_entry.get('compression_ratio', 0.0)
        }

class SensitivityAnalyzer:
    def __init__(self, c_max: float = 0.9999, c_min: float = 0.99):
        self.c_max = c_max
        self.c_min = c_min
        self.sigma_values = [1.0, 5.0, 10.0, 20.0, 50.0]
        
    def analyze_compression_sensitivity(self, delta_p_range: List[float] = None) -> Dict[str, Any]:
        if delta_p_range is None:
            delta_p_range = np.linspace(0, 0.3, 31)
        
        results = {}
        
        for sigma in self.sigma_values:
            compression_ratios = []
            sensitivities = []
            
            for delta_p in delta_p_range:
                compression_ratio = self.calculate_compression_ratio(delta_p, sigma)
                sensitivity = self.calculate_sensitivity(delta_p, sigma)
                
                compression_ratios.append(compression_ratio)
                sensitivities.append(sensitivity)
            
            results[f'sigma_{sigma}'] = {
                'delta_p': delta_p_range.tolist(),
                'compression_ratios': compression_ratios,
                'sensitivities': sensitivities
            }
        
        return results
    
    def calculate_compression_ratio(self, delta_p: float, sigma: float) -> float:
        compression_ratio = self.c_max - (2 * (self.c_max - self.c_min)) / (1 + np.exp(delta_p * sigma))
        return max(self.c_min, min(self.c_max, compression_ratio))
    
    def calculate_sensitivity(self, delta_p: float, sigma: float) -> float:
        exp_term = np.exp(delta_p * sigma)
        sensitivity = 2 * (self.c_max - self.c_min) * sigma * exp_term / ((1 + exp_term) ** 2)
        return sensitivity
    
    def plot_sensitivity_analysis(self, results: Dict[str, Any], save_path: str = 'sensitivity_analysis.png'):
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            colors = ['blue', 'green', 'red', 'orange', 'purple']
            
            for i, sigma in enumerate(self.sigma_values):
                key = f'sigma_{sigma}'
                if key in results:
                    delta_p = results[key]['delta_p']
                    compression_ratios = [c * 100 for c in results[key]['compression_ratios']]
                    sensitivities = results[key]['sensitivities']
                    
                    ax1.plot(delta_p, compression_ratios, color=colors[i], 
                            label=f'σ={sigma}', linewidth=2)
                    ax2.plot(delta_p, sensitivities, color=colors[i], 
                            label=f'σ={sigma}', linewidth=2)
            
            ax1.set_xlabel('Performance Fluctuation (Δp)')
            ax1.set_ylabel('Compression Ratio C (%)')
            ax1.set_title('Sensitivity Analysis: Compression Ratio vs Performance Fluctuation')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(98, 100)
            
            ax2.set_xlabel('Performance Fluctuation (Δp)')
            ax2.set_ylabel('Sensitivity |dC/dΔp|')
            ax2.set_title('Sensitivity Analysis: Compression Sensitivity vs Performance Fluctuation')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Sensitivity analysis plot saved to {save_path}")
            
        except ImportError:
            print("Matplotlib not available for plotting sensitivity analysis")
    
    def generate_sensitivity_report(self, results: Dict[str, Any]) -> str:
        report = "Dynamic Weight Compression Sensitivity Analysis Report\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Parameters:\n"
        report += f"- C_max: {self.c_max:.4f} ({self.c_max*100:.2f}%)\n"
        report += f"- C_min: {self.c_min:.4f} ({self.c_min*100:.2f}%)\n"
        report += f"- σ values tested: {self.sigma_values}\n\n"
        
        report += "Key Findings:\n"
        
        for sigma in self.sigma_values:
            key = f'sigma_{sigma}'
            if key in results:
                delta_p_01 = 0.1
                compression_at_01 = self.calculate_compression_ratio(delta_p_01, sigma)
                sensitivity_at_01 = self.calculate_sensitivity(delta_p_01, sigma)
                
                report += f"\nσ = {sigma}:\n"
                report += f"- C at Δp=0.1: {compression_at_01:.4f} ({compression_at_01*100:.2f}%)\n"
                report += f"- Sensitivity at Δp=0.1: {sensitivity_at_01:.4f}\n"
                report += f"- Adjustment rate: ~{sensitivity_at_01*100:.2f}% per 0.1 Δp\n"
        
        report += f"\nRecommended σ = 10.0:\n"
        report += f"- Balanced adaptation for typical Δp ranges (0.01-0.10)\n"
        report += f"- Practical adjustment rate of ~0.5% per 0.1 Δp\n"
        report += f"- Suitable for both early training volatility and stable convergence\n"
        
        return report

class AdaptiveCompressionScheduler:
    def __init__(self, initial_c_max: float = 0.9999, initial_c_min: float = 0.99, 
                 adaptation_rate: float = 0.1, performance_window: int = 5):
        self.c_max = initial_c_max
        self.c_min = initial_c_min
        self.adaptation_rate = adaptation_rate
        self.performance_window = performance_window
        self.performance_history = {}
        
    def update_compression_bounds(self, client_id: int, recent_performance: List[float]):
        if len(recent_performance) < self.performance_window:
            return
        
        performance_trend = np.mean(recent_performance[-self.performance_window:]) - np.mean(recent_performance[:-self.performance_window])
        
        if performance_trend > 0:
            self.c_min = max(0.95, self.c_min - self.adaptation_rate * 0.01)
            self.c_max = max(0.999, self.c_max - self.adaptation_rate * 0.001)
        else:
            self.c_min = min(0.995, self.c_min + self.adaptation_rate * 0.01)
            self.c_max = min(0.9999, self.c_max + self.adaptation_rate * 0.001)
    
    def get_compression_bounds(self) -> Tuple[float, float]:
        return self.c_max, self.c_min

class CompressionAnalyzer:
    def __init__(self):
        self.compression_history = []
        self.performance_history = []
        
    def record_compression(self, client_id: int, compression_ratio: float, 
                          accuracy: float, loss: float, communication_reduction: float, delta_p: float = None):
        self.compression_history.append({
            'client_id': client_id,
            'compression_ratio': compression_ratio,
            'accuracy': accuracy,
            'loss': loss,
            'communication_reduction': communication_reduction,
            'delta_p': delta_p,
            'timestamp': len(self.compression_history)
        })
    
    def get_compression_analysis(self) -> Dict[str, Any]:
        if not self.compression_history:
            return {}
        
        compression_ratios = [entry['compression_ratio'] for entry in self.compression_history]
        accuracies = [entry['accuracy'] for entry in self.compression_history]
        communication_reductions = [entry['communication_reduction'] for entry in self.compression_history]
        delta_ps = [entry.get('delta_p', 0) for entry in self.compression_history]
        
        return {
            'avg_compression_ratio': np.mean(compression_ratios),
            'avg_accuracy': np.mean(accuracies),
            'avg_communication_reduction': np.mean(communication_reductions),
            'avg_delta_p': np.mean(delta_ps),
            'compression_ratio_std': np.std(compression_ratios),
            'accuracy_std': np.std(accuracies),
            'total_rounds': len(self.compression_history)
        }
    
    def plot_compression_trends(self):
        if not self.compression_history:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            rounds = [entry['timestamp'] for entry in self.compression_history]
            compression_ratios = [entry['compression_ratio'] * 100 for entry in self.compression_history]
            accuracies = [entry['accuracy'] for entry in self.compression_history]
            delta_ps = [entry.get('delta_p', 0) for entry in self.compression_history]
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            
            ax1.plot(rounds, compression_ratios, 'b-', label='Compression Ratio')
            ax1.set_xlabel('Training Rounds')
            ax1.set_ylabel('Compression Ratio (%)')
            ax1.set_title('Dynamic Compression Ratio Over Time')
            ax1.legend()
            ax1.grid(True)
            ax1.set_ylim(98, 100)
            
            ax2.plot(rounds, accuracies, 'r-', label='Accuracy')
            ax2.set_xlabel('Training Rounds')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Model Accuracy Over Time')
            ax2.legend()
            ax2.grid(True)
            
            ax3.plot(rounds, delta_ps, 'g-', label='Performance Fluctuation (Δp)')
            ax3.set_xlabel('Training Rounds')
            ax3.set_ylabel('Δp')
            ax3.set_title('Performance Fluctuation Over Time')
            ax3.legend()
            ax3.grid(True)
            
            plt.tight_layout()
            plt.savefig('compression_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for plotting")
