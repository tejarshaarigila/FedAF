# utils_fedavg.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

def load_data(dataset, alpha, num_clients):
    """Load and partition data among clients using Dirichlet distribution for non-IID setting."""
    logger.info("Loading %s dataset with alpha: %.4f", dataset, alpha)
    if dataset == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        full_train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    elif dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        full_train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    elif dataset == 'CelebA':
        transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        full_train_dataset = datasets.CelebA(root='data', split='train', download=True, transform=transform)
        test_dataset = datasets.CelebA(root='data', split='test', download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset.")

    # Split data into non-IID partitions
    targets = np.array(full_train_dataset.targets)
    client_indices = partition_data(targets, num_clients, alpha)
    logger.info("Data partitioned among %d clients.", num_clients)

    client_datasets = [Subset(full_train_dataset, indices) for indices in client_indices]
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    return client_datasets, test_loader

def partition_data( targets, num_clients, alpha):
    """Partition data using Dirichlet distribution."""
    np.random.seed(42)
    client_indices = [[] for _ in range(num_clients)]
    class_counts = np.bincount(targets)
    for class_idx in range(len(class_counts)):
        class_indices = np.where(targets == class_idx)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        client_splits = np.split(class_indices, proportions)
        for i, client_split in enumerate(client_splits):
            client_indices[i].extend(client_split)

    logger.info("Data partitioning complete.")
    return client_indices
