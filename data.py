import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import random
from torch.utils.data import random_split
import os

def get_mnist_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset

def get_fashionmnist_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset

def get_cifar10_datasets():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    return train_dataset, test_dataset

def get_clients_datasets(train_dataset, num_clients):
    num_samples = len(train_dataset)
    samples_per_client = num_samples // num_clients
    
    client_datasets = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else num_samples
        client_dataset = Subset(train_dataset, range(start_idx, end_idx))
        client_datasets.append(client_dataset)
    
    return client_datasets

def get_fashionmnist_clients_datasets(train_dataset, num_clients):
    return get_clients_datasets(train_dataset, num_clients)

def get_CIFAR10(dir_alpha, num_clients):
    train_dataset, test_dataset = get_cifar10_datasets()
    
    if dir_alpha > 0:
        client_datasets = partition_data_dirichlet(train_dataset, num_clients, dir_alpha)
    else:
        client_datasets = get_clients_datasets(train_dataset, num_clients)
    
    client_data_sizes = [len(client_dataset) for client_dataset in client_datasets]
    clients_train_loaders = [DataLoader(client_dataset, batch_size=64, shuffle=True) for client_dataset in client_datasets]
    clients_test_loaders = [DataLoader(test_dataset, batch_size=64, shuffle=False) for i in range(num_clients)]
    
    return clients_train_loaders, clients_test_loaders, client_data_sizes

def partition_data_dirichlet(dataset, num_clients, alpha):
    min_size = 0
    N = len(dataset)
    client_data_idx = {}
    
    while min_size < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(10):
            idx_k = np.where(np.array(dataset.targets) == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = np.array([p * (len(idx_k)) for p in proportions])
            proportions = np.array([int(p) for p in proportions])
            proportions[0] += len(idx_k) - sum(proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    
    client_datasets = []
    for j in range(num_clients):
        client_datasets.append(Subset(dataset, idx_batch[j]))
    
    return client_datasets

def create_data_loaders(dataset_name, num_clients, batch_size=64, dir_alpha=100):
    if dataset_name == 'MNIST':
        train_dataset, test_dataset = get_mnist_datasets()
        client_datasets = get_clients_datasets(train_dataset, num_clients)
    elif dataset_name == 'FashionMNIST':
        train_dataset, test_dataset = get_fashionmnist_datasets()
        client_datasets = get_fashionmnist_clients_datasets(train_dataset, num_clients)
    elif dataset_name == 'CIFAR10':
        train_dataset, test_dataset = get_cifar10_datasets()
        if dir_alpha > 0:
            client_datasets = partition_data_dirichlet(train_dataset, num_clients, dir_alpha)
        else:
            client_datasets = get_clients_datasets(train_dataset, num_clients)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    client_data_sizes = [len(client_dataset) for client_dataset in client_datasets]
    clients_train_loaders = [DataLoader(client_dataset, batch_size=batch_size, shuffle=True) for client_dataset in client_datasets]
    clients_test_loaders = [DataLoader(test_dataset, batch_size=batch_size, shuffle=False) for i in range(num_clients)]
    
    return clients_train_loaders, clients_test_loaders, client_data_sizes, client_datasets

def download_dataset(dataset_name):
    print(f"Downloading {dataset_name} dataset...")
    
    if dataset_name == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    elif dataset_name == 'FashionMNIST':
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True)
    elif dataset_name == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    print(f"{dataset_name} dataset downloaded successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset

def check_dataset_availability(dataset_name):
    dataset_path = f'./data/{dataset_name.lower()}'
    if dataset_name == 'CIFAR10':
        dataset_path = './data/cifar-10-batches-py'
    
    if os.path.exists(dataset_path):
        print(f"{dataset_name} dataset found at {dataset_path}")
        return True
    else:
        print(f"{dataset_name} dataset not found at {dataset_path}")
        return False

def ensure_dataset_available(dataset_name):
    if not check_dataset_availability(dataset_name):
        download_dataset(dataset_name)
    else:
        print(f"{dataset_name} dataset is already available.")




