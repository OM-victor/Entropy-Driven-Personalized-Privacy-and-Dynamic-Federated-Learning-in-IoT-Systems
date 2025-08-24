import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
import math

class EntropyCalculator:
    def __init__(self):
        pass
    
    def calculate_client_entropy(self, dataset, num_classes: int) -> float:
        labels = []
        for _, label in dataset:
            labels.append(label.item() if torch.is_tensor(label) else label)
        
        label_counts = Counter(labels)
        total_samples = len(labels)
        
        entropy = 0.0
        for class_id in range(num_classes):
            count = label_counts.get(class_id, 0)
            if count > 0:
                probability = count / total_samples
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def calculate_all_clients_entropy(self, clients_datasets: List, num_classes: int) -> List[float]:
        entropies = []
        for client_dataset in clients_datasets:
            entropy = self.calculate_client_entropy(client_dataset, num_classes)
            entropies.append(entropy)
        return entropies
    
    def get_entropy_statistics(self, entropies: List[float]) -> Dict[str, float]:
        return {
            'mean': np.mean(entropies),
            'std': np.std(entropies),
            'min': np.min(entropies),
            'max': np.max(entropies),
            'median': np.median(entropies)
        }
    
    def normalize_entropies(self, entropies: List[float]) -> List[float]:
        total_entropy = sum(entropies)
        if total_entropy == 0:
            return [1.0 / len(entropies)] * len(entropies)
        return [entropy / total_entropy for entropy in entropies]
