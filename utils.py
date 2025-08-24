import torch
from opacus.accountants.utils import get_noise_multiplier
from options import parse_args
from torch import autograd

args = parse_args()


def compute_noise_multiplier(target_epsilon, target_delta, global_epoch, local_epoch, batch_size, client_data_sizes):
    total_dataset_size = sum(client_data_sizes)
    sample_rate = batch_size / total_dataset_size * args.user_sample_rate
    total_steps = args.user_sample_rate * sum(global_epoch * local_epoch * (client_data_size / batch_size) for client_data_size in client_data_sizes)

    return get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        steps=total_steps, 
        accountant="rdp"
    )


def compute_fisher_diag(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    fisher_diag = [torch.zeros_like(param) for param in model.parameters()]
    num_samples = 0
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            batch_size = data.size(0)
            num_samples += batch_size
            
            log_probs = torch.nn.functional.log_softmax(model(data), dim=1)
            
            for i, label in enumerate(labels):
                log_prob = log_probs[i, label]
                
                model.zero_grad()
                grad1 = autograd.grad(log_prob, model.parameters(), create_graph=True, retain_graph=True)
                
                for fisher_diag_value, grad_value in zip(fisher_diag, grad1):
                    fisher_diag_value.add_(grad_value.detach() ** 2)
                
                del log_prob, grad1
    
    fisher_diag = [fisher_diag_value / num_samples for fisher_diag_value in fisher_diag]
    
    normalized_fisher_diag = []
    for fisher_value in fisher_diag:
        x_min = torch.min(fisher_value)
        x_max = torch.max(fisher_value)
        denominator = x_max - x_min
        if denominator > 1e-8:
            normalized_fisher_value = (fisher_value - x_min) / denominator
        else:
            normalized_fisher_value = fisher_value - x_min
        normalized_fisher_diag.append(normalized_fisher_value)
    
    return normalized_fisher_diag
    