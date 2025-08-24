import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class ParameterMasks:
    mask_1: torch.Tensor
    mask_2: torch.Tensor
    significant_params: List[torch.Tensor]
    non_significant_params: List[torch.Tensor]

class InformationValueCalculator:
    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold
        
    def compute_information_values(self, model: nn.Module, dataloader, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        model.eval()
        information_values = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                information_values[name] = torch.zeros_like(param.data)
        
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            total_samples += batch_size
            
            model.zero_grad()
            
            output = model(data)
            log_probs = F.log_softmax(output, dim=1)
            
            log_likelihood = torch.sum(log_probs.gather(1, target.unsqueeze(1)))
            log_likelihood.backward()
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    information_values[name] += param.grad.data ** 2
        
        for name in information_values:
            information_values[name] /= total_samples
            
        return information_values
    
    def normalize_information_values(self, information_values: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalized_values = {}
        
        for name, values in information_values.items():
            min_val = torch.min(values)
            max_val = torch.max(values)
            
            if max_val > min_val:
                normalized_values[name] = (values - min_val) / (max_val - min_val)
            else:
                normalized_values[name] = torch.ones_like(values) * 0.5
                
        return normalized_values

class DynamicParameterPartitioner:
    def __init__(self, threshold: float = 0.4, lambda_1: float = 0.1, lambda_2: float = 0.05):
        self.threshold = threshold
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.information_calculator = InformationValueCalculator(threshold)
        
    def partition_parameters(self, model: nn.Module, dataloader, device: str = 'cpu') -> ParameterMasks:
        information_values = self.information_calculator.compute_information_values(model, dataloader, device)
        
        normalized_values = self.information_calculator.normalize_information_values(information_values)
        
        mask_1 = {}
        mask_2 = {}
        significant_params = []
        non_significant_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                normalized_value = normalized_values[name]
                
                significant_mask = (normalized_value > self.threshold).float()
                non_significant_mask = (normalized_value <= self.threshold).float()
                
                mask_1[name] = significant_mask
                mask_2[name] = non_significant_mask
                
                significant_param = param.data * significant_mask
                non_significant_param = param.data * non_significant_mask
                
                significant_params.append(significant_param)
                non_significant_params.append(non_significant_param)
        
        return ParameterMasks(
            mask_1=mask_1,
            mask_2=mask_2,
            significant_params=significant_params,
            non_significant_params=non_significant_params
        )
    
    def apply_parameter_update(self, local_model: nn.Module, global_model: nn.Module, 
                             masks: ParameterMasks) -> nn.Module:
        local_state = local_model.state_dict()
        global_state = global_model.state_dict()
        
        updated_state = {}
        param_idx = 0
        
        for name, param in local_model.named_parameters():
            if param.requires_grad:
                mask_1 = masks.mask_1[name]
                mask_2 = masks.mask_2[name]
                
                local_param = local_state[name]
                global_param = global_state[name]
                
                updated_param = mask_1 * local_param + mask_2 * global_param
                updated_state[name] = updated_param
                param_idx += 1
            else:
                updated_state[name] = param.data
        
        local_model.load_state_dict(updated_state)
        return local_model

class DifferentiatedRegularization:
    def __init__(self, lambda_1: float = 0.1, lambda_2: float = 0.05, clipping_bound: float = 1.0):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.clipping_bound = clipping_bound
        
    def compute_loss_j1(self, model: nn.Module, data: torch.Tensor, target: torch.Tensor, 
                       significant_params: List[torch.Tensor], previous_significant: List[torch.Tensor]) -> torch.Tensor:
        output = model(data)
        cross_entropy_loss = F.cross_entropy(output, target)
        
        l1_norm = 0.0
        l2_norm = 0.0
        
        for current_param, prev_param in zip(significant_params, previous_significant):
            diff = current_param - prev_param
            l1_norm += torch.norm(diff, p=1)
            l2_norm += torch.norm(diff, p=2) ** 2
        
        regularization_loss = self.lambda_1 * (0.5 * l1_norm + 0.25 * l2_norm)
        total_loss = cross_entropy_loss + regularization_loss
        
        return total_loss
    
    def compute_loss_j2(self, model: nn.Module, data: torch.Tensor, target: torch.Tensor,
                       non_significant_params: List[torch.Tensor], previous_non_significant: List[torch.Tensor]) -> torch.Tensor:
        output = model(data)
        cross_entropy_loss = F.cross_entropy(output, target)
        
        l2_constraint_loss = 0.0
        
        for current_param, prev_param in zip(non_significant_params, previous_non_significant):
            diff = current_param - prev_param
            l2_norm = torch.norm(diff, p=2)
            constraint = torch.relu(l2_norm - self.clipping_bound)
            l2_constraint_loss += constraint ** 2
        
        regularization_loss = 0.5 * self.lambda_2 * l2_constraint_loss
        total_loss = cross_entropy_loss + regularization_loss
        
        return total_loss
    
    def update_significant_parameters(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                                    dataloader, masks: ParameterMasks, device: str = 'cpu'):
        model.train()
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            current_significant = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    significant_param = param.data * masks.mask_1[name]
                    current_significant.append(significant_param)
            
            loss = self.compute_loss_j1(model, data, target, current_significant, current_significant)
            
            loss.backward()
            optimizer.step()
    
    def update_non_significant_parameters(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                                        dataloader, masks: ParameterMasks, device: str = 'cpu'):
        model.train()
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            current_non_significant = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    non_significant_param = param.data * masks.mask_2[name]
                    current_non_significant.append(non_significant_param)
            
            loss = self.compute_loss_j2(model, data, target, current_non_significant, current_non_significant)
            
            loss.backward()
            optimizer.step()

class ClientLocalTraining:
    def __init__(self, threshold: float = 0.4, lambda_1: float = 0.1, lambda_2: float = 0.05, 
                 clipping_bound: float = 1.0, learning_rate: float = 0.01):
        self.parameter_partitioner = DynamicParameterPartitioner(threshold, lambda_1, lambda_2)
        self.regularization = DifferentiatedRegularization(lambda_1, lambda_2, clipping_bound)
        self.learning_rate = learning_rate
        
    def train_local_model(self, local_model: nn.Module, global_model: nn.Module, 
                         dataloader, num_epochs: int, device: str = 'cpu') -> Tuple[nn.Module, float]:
        local_model.to(device)
        global_model.to(device)
        
        total_loss = 0.0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            masks = self.parameter_partitioner.partition_parameters(local_model, dataloader, device)
            
            local_model = self.parameter_partitioner.apply_parameter_update(local_model, global_model, masks)
            
            optimizer = torch.optim.SGD(local_model.parameters(), lr=self.learning_rate)
            
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                
                self.regularization.update_significant_parameters(
                    local_model, optimizer, [(data, target)], masks, device
                )
                
                self.regularization.update_non_significant_parameters(
                    local_model, optimizer, [(data, target)], masks, device
                )
                
                with torch.no_grad():
                    output = local_model(data)
                    loss = F.cross_entropy(output, target)
                    epoch_loss += loss.item()
            
            total_loss += epoch_loss / len(dataloader)
        
        return local_model, total_loss / num_epochs
    
    def get_parameter_masks_info(self, model: nn.Module, dataloader, device: str = 'cpu') -> Dict[str, Any]:
        masks = self.parameter_partitioner.partition_parameters(model, dataloader, device)
        
        significant_count = 0
        non_significant_count = 0
        
        for name, mask_1 in masks.mask_1.items():
            significant_count += torch.sum(mask_1).item()
            non_significant_count += torch.sum(masks.mask_2[name]).item()
        
        return {
            'significant_parameters': significant_count,
            'non_significant_parameters': non_significant_count,
            'total_parameters': significant_count + non_significant_count,
            'significant_ratio': significant_count / (significant_count + non_significant_count)
        }