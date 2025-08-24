import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from dynamic_compression import DynamicWeightCompression
from personalized_dp import PersonalizedDifferentialPrivacy
from entropy_calculator import calculate_entropy
from algorithm_implementation import ClientLocalTraining
from options import parse_args
import numpy as np
import copy

args = parse_args()

class FederatedClient:
    def __init__(self, client_id: int, model: nn.Module, train_loader, test_loader, data_size: int, dataset=None):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.data_size = data_size
        self.dataset = dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.local_trainer = ClientLocalTraining(
            threshold=args.fisher_threshold,
            lambda_1=args.lambda_1,
            lambda_2=args.lambda_2,
            clipping_bound=args.clipping_bound,
            learning_rate=args.lr
        )
        
        self.personalized_dp = PersonalizedDifferentialPrivacy(
            total_epsilon=args.target_epsilon,
            delta=args.target_delta,
            clipping_bound=args.clipping_bound
        )
        
        self.dynamic_compression = DynamicWeightCompression(
            c_max=args.c_max if hasattr(args, 'c_max') else 0.9999,
            c_min=args.c_min if hasattr(args, 'c_min') else 0.99,
            sigma=args.compression_sigma if hasattr(args, 'compression_sigma') else 10.0,
            w_a=args.w_a if hasattr(args, 'w_a') else 0.5,
            w_b=args.w_b if hasattr(args, 'w_b') else 0.5
        )
        
        self.residual = None
        self.performance_history = []
        self.previous_model_state = None
        
    def train_local_model(self, global_model_state: Dict[str, torch.Tensor], local_epochs: int) -> Tuple[List[torch.Tensor], float]:
        if self.previous_model_state is None:
            self.previous_model_state = copy.deepcopy(self.model.state_dict())
        
        global_model = type(self.model)()
        global_model.load_state_dict(global_model_state)
        global_model.to(self.device)
        
        trained_model, total_loss = self.local_trainer.train_local_model(
            self.model, global_model, self.train_loader, local_epochs, self.device
        )
        
        self.model = trained_model
        
        model_update = self._compute_model_update(global_model_state)
        
        self.previous_model_state = copy.deepcopy(self.model.state_dict())
        
        return model_update, total_loss
    
    def _compute_model_update(self, global_model_state: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        model_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                global_param = global_model_state[name]
                update = param.data - global_param
                model_update.append(update)
        return model_update
    
    def apply_personalized_differential_privacy(self, model_update: List[torch.Tensor]) -> List[torch.Tensor]:
        if args.no_noise:
            return model_update
        
        if self.dataset is not None:
            entropy = calculate_entropy(self.dataset)
            return self.personalized_dp.apply_noise_with_entropy(model_update, entropy, self.client_id)
        else:
            return self.personalized_dp.apply_noise(model_update, self.client_id)
    
    def _apply_residual(self, model_update: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.residual is None:
            self.residual = [torch.zeros_like(update) for update in model_update]
            return model_update
        
        for i, (update, res) in enumerate(zip(model_update, self.residual)):
            model_update[i] = update + res
            self.residual[i] = torch.zeros_like(res)
        
        return model_update
    
    def compress_update(self, model_update: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if not args.compression:
            return model_update, [torch.zeros_like(update) for update in model_update]
        
        compressed_update = []
        residual_update = []
        
        for param_update in model_update:
            compressed_param = torch.sign(param_update) * torch.sqrt(torch.abs(param_update))
            residual = param_update - compressed_param
            compressed_update.append(compressed_param)
            residual_update.append(residual)
        
        return compressed_update, residual_update
    
    def calculate_residual(self, model_update: List[torch.Tensor], compressed_update: List[torch.Tensor]) -> List[torch.Tensor]:
        residual = []
        for param_update, comp_update in zip(model_update, compressed_update):
            res = param_update - comp_update
            residual.append(res)
        return residual
    
    def evaluate_model_with_loss(self) -> Tuple[float, float]:
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        loss = test_loss / total
        
        return accuracy, loss
    
    def apply_dynamic_compression(self, model_update: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], float, float]:
        accuracy, loss = self.evaluate_model_with_loss()
        
        compressed_update, residual_update, compression_ratio = self.dynamic_compression.apply_dynamic_compression(
            self.client_id, model_update, accuracy, loss
        )
        
        delta_p = self.dynamic_compression.calculate_performance_change(
            self.dynamic_compression.load_previous_performance(self.client_id)[0],
            self.dynamic_compression.load_previous_performance(self.client_id)[1],
            accuracy, loss
        )
        
        return compressed_update, residual_update, compression_ratio, delta_p
    
    def get_parameter_partitioning_info(self) -> Dict[str, Any]:
        return self.local_trainer.get_parameter_masks_info(self.model, self.train_loader, self.device)
    
    def get_privacy_info(self) -> Dict[str, Any]:
        if self.dataset is not None:
            entropy = calculate_entropy(self.dataset)
            return {
                'client_id': self.client_id,
                'entropy': entropy,
                'epsilon': self.personalized_dp.get_client_epsilon(entropy, self.client_id)
            }
        return {'client_id': self.client_id, 'entropy': 0.0, 'epsilon': 0.0}
    
    def get_compression_info(self) -> Dict[str, Any]:
        return self.dynamic_compression.get_compression_stats(self.client_id)
    
    def update_model(self, global_model_state: Dict[str, torch.Tensor]):
        self.model.load_state_dict(global_model_state)
