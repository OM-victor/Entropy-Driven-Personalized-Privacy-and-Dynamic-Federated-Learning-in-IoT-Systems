import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from client import FederatedClient
from personalized_dp import PersonalizedDifferentialPrivacy, AdaptivePersonalizedDP
from sensitivity_analysis import SensitivityAnalyzer
from options import parse_args
import numpy as np
import copy
import os

args = parse_args()

class FederatedServer:
    def __init__(self, global_model: nn.Module, clients: List[FederatedClient], num_classes: int):
        self.global_model = global_model
        self.clients = clients
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model.to(self.device)
        
        if args.adaptive_dp:
            self.personalized_dp = AdaptivePersonalizedDP(
                total_epsilon=args.target_epsilon,
                delta=args.target_delta,
                clipping_bound=args.clipping_bound,
                adaptation_rate=args.adaptation_rate
            )
        else:
            self.personalized_dp = PersonalizedDifferentialPrivacy(
                total_epsilon=args.target_epsilon,
                delta=args.target_delta,
                clipping_bound=args.clipping_bound
            )
        
        if args.sensitivity_analysis:
            self.sensitivity_analyzer = SensitivityAnalyzer(
                c_max=args.c_max if hasattr(args, 'c_max') else 0.9999,
                c_min=args.c_min if hasattr(args, 'c_min') else 0.99
            )
        
        self.training_history = {
            'global_accuracies': [],
            'client_accuracies': [],
            'privacy_info': [],
            'compression_info': [],
            'parameter_partitioning_info': []
        }
        
        self._initialize_privacy_budgets()
    
    def _initialize_privacy_budgets(self):
        client_datasets = [client.dataset for client in self.clients if client.dataset is not None]
        if client_datasets:
            self.personalized_dp.allocate_privacy_budgets(client_datasets)
    
    def federated_learning(self) -> Dict[str, Any]:
        print("Starting federated learning with integrated framework...")
        print("Framework components:")
        print("- Dynamic parameter partitioning with information value-based categorization")
        print("- Differentiated regularization (Elastic net for significant, L2 constraint for non-significant)")
        print("- Adaptive weight compression with performance feedback")
        print("- Entropy-driven personalized differential privacy")
        
        if args.sensitivity_analysis:
            print("Running sensitivity analysis...")
            sensitivity_results = self.sensitivity_analyzer.analyze_compression_sensitivity()
            self.sensitivity_analyzer.plot_sensitivity_analysis(sensitivity_results)
            sensitivity_report = self.sensitivity_analyzer.generate_sensitivity_report(sensitivity_results)
            print(sensitivity_report)
        
        for epoch in range(args.global_epoch):
            print(f"\nGlobal Epoch {epoch + 1}/{args.global_epoch}")
            
            global_model_state = copy.deepcopy(self.global_model.state_dict())
            
            sampled_clients = self._sample_clients()
            client_updates = []
            epoch_privacy_info = []
            epoch_compression_info = []
            epoch_partitioning_info = []
            
            for client in sampled_clients:
                print(f"Training client {client.client_id}...")
                
                
                model_update, train_loss = client.train_local_model(global_model_state, args.local_epoch)
                
                
                partitioning_info = client.get_parameter_partitioning_info()
                epoch_partitioning_info.append(partitioning_info)
                
                print(f"Client {client.client_id} - Parameter partitioning:")
                print(f"  Significant parameters: {partitioning_info['significant_parameters']}")
                print(f"  Non-significant parameters: {partitioning_info['non_significant_parameters']}")
                print(f"  Significant ratio: {partitioning_info['significant_ratio']:.3f}")
                
                if args.dynamic_compression:
                    compressed_update, residual_update, compression_ratio, delta_p = client.apply_dynamic_compression(model_update)
                    
                    compression_info = {
                        'client_id': client.client_id,
                        'compression_ratio': compression_ratio,
                        'communication_reduction': 1.0 - compression_ratio,
                        'delta_p': delta_p,
                        'transmitted_params_ratio': compression_ratio * 100
                    }
                    epoch_compression_info.append(compression_info)
                    
                    print(f"Client {client.client_id} - Compression ratio: {compression_ratio:.4f} ({compression_ratio*100:.2f}%), "
                          f"Δp: {delta_p:.4f}, Communication reduction: {1.0 - compression_ratio:.4f}")
                    
                    model_update = compressed_update
                elif args.compression:
                    compressed_update, residual_update = client.compress_update(model_update)
                    client.calculate_residual(model_update, compressed_update)
                    model_update = compressed_update
                
               
                private_update = client.apply_personalized_differential_privacy(model_update)
                
                privacy_info = client.get_privacy_info()
                epoch_privacy_info.append(privacy_info)
                
                print(f"Client {client.client_id} - Privacy: ε={privacy_info['epsilon']:.3f}, "
                      f"Entropy={privacy_info['entropy']:.3f}")
                
                client_updates.append({
                    'client_id': client.client_id,
                    'update': private_update,
                    'data_size': client.data_size
                })
            
           
            self._aggregate_updates(client_updates)
            
            
            global_accuracy = self._evaluate_global_model()
            client_accuracies = self._evaluate_client_models()
            
            
            self.training_history['global_accuracies'].append(global_accuracy)
            self.training_history['client_accuracies'].append(client_accuracies)
            self.training_history['privacy_info'].append(epoch_privacy_info)
            self.training_history['compression_info'].append(epoch_compression_info)
            self.training_history['parameter_partitioning_info'].append(epoch_partitioning_info)
            
            print(f"Global accuracy: {global_accuracy:.2f}%")
            print(f"Average client accuracy: {np.mean(client_accuracies):.2f}%")
            
            
            if args.adaptive_dp:
                self.personalized_dp.update_privacy_budgets(epoch_privacy_info, global_accuracy)
        
        return self.training_history
    
    def _sample_clients(self) -> List[FederatedClient]:
        num_sampled = max(1, int(args.num_clients * args.user_sample_rate))
        sampled_indices = np.random.choice(len(self.clients), num_sampled, replace=False)
        return [self.clients[i] for i in sampled_indices]
    
    def _aggregate_updates(self, client_updates: List[Dict[str, Any]]):
        total_data_size = sum(update['data_size'] for update in client_updates)
        
        aggregated_update = []
        for i, param in enumerate(self.global_model.parameters()):
            if param.requires_grad:
                weighted_update = torch.zeros_like(param.data)
                
                for update_info in client_updates:
                    weight = update_info['data_size'] / total_data_size
                    weighted_update += weight * update_info['update'][i]
                
                param.data += weighted_update
    
    def _evaluate_global_model(self) -> float:
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for client in self.clients:
                for data, target in client.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.global_model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                    break  # Only evaluate on first batch per client
        
        return 100. * correct / total if total > 0 else 0.0
    
    def _evaluate_client_models(self) -> List[float]:
        client_accuracies = []
        for client in self.clients:
            accuracy, _ = client.evaluate_model_with_loss()
            client_accuracies.append(accuracy)
        return client_accuracies
    
    def get_privacy_statistics(self) -> Dict[str, Any]:
        all_epsilons = []
        all_entropies = []
        
        for epoch_info in self.training_history['privacy_info']:
            for client_info in epoch_info:
                all_epsilons.append(client_info['epsilon'])
                all_entropies.append(client_info['entropy'])
        
        return {
            'statistics': {
                'avg_entropy': np.mean(all_entropies),
                'std_entropy': np.std(all_entropies),
                'min_entropy': np.min(all_entropies),
                'max_entropy': np.max(all_entropies)
            },
            'epsilons': all_epsilons,
            'entropies': all_entropies
        }
    
    def get_compression_statistics(self) -> Dict[str, Any]:
        all_compression_ratios = []
        all_communication_reductions = []
        all_delta_ps = []
        all_transmitted_params = []
        
        for epoch_info in self.training_history['compression_info']:
            for client_info in epoch_info:
                all_compression_ratios.append(client_info['compression_ratio'])
                all_communication_reductions.append(client_info['communication_reduction'])
                all_delta_ps.append(client_info['delta_p'])
                all_transmitted_params.append(client_info['transmitted_params_ratio'])
        
        return {
            'avg_compression_ratio': np.mean(all_compression_ratios),
            'avg_communication_reduction': np.mean(all_communication_reductions),
            'avg_delta_p': np.mean(all_delta_ps),
            'avg_transmitted_params_ratio': np.mean(all_transmitted_params),
            'compression_ratio_std': np.std(all_compression_ratios),
            'delta_p_std': np.std(all_delta_ps),
            'total_compression_rounds': len(all_compression_ratios),
            'compression_ratio_range': [np.min(all_compression_ratios), np.max(all_compression_ratios)],
            'delta_p_range': [np.min(all_delta_ps), np.max(all_delta_ps)]
        }
    
    def get_parameter_partitioning_statistics(self) -> Dict[str, Any]:
        all_significant_ratios = []
        all_significant_counts = []
        all_non_significant_counts = []
        
        for epoch_info in self.training_history['parameter_partitioning_info']:
            for client_info in epoch_info:
                all_significant_ratios.append(client_info['significant_ratio'])
                all_significant_counts.append(client_info['significant_parameters'])
                all_non_significant_counts.append(client_info['non_significant_parameters'])
        
        return {
            'avg_significant_ratio': np.mean(all_significant_ratios),
            'avg_significant_count': np.mean(all_significant_counts),
            'avg_non_significant_count': np.mean(all_non_significant_counts),
            'significant_ratio_std': np.std(all_significant_ratios),
            'total_partitioning_rounds': len(all_significant_ratios),
            'significant_ratio_range': [np.min(all_significant_ratios), np.max(all_significant_ratios)]
        }
    
    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        privacy_stats = self.get_privacy_statistics()
        compression_stats = self.get_compression_statistics() if args.dynamic_compression else {}
        partitioning_stats = self.get_parameter_partitioning_statistics()
        
        checkpoint = {
            'global_model_state': self.global_model.state_dict(),
            'training_history': self.training_history,
            'privacy_info': privacy_stats,
            'compression_info': compression_stats,
            'parameter_partitioning_info': partitioning_stats,
            'args': vars(args)
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.global_model.load_state_dict(checkpoint['global_model_state'])
        self.training_history = checkpoint['training_history']
