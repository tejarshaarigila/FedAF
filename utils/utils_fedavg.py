# utils_fedavg.py

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import logging
import time
from utils.networks import (
    MLP, ConvNet, LeNet, AlexNet, AlexNetBN, VGG11, VGG11BN,
    ResNet18, ResNet18BN_AP, ResNet18BN
)

# Configure logging
logger = logging.getLogger(__name__)

def load_data(dataset, alpha, num_clients, seed=42):
    """Load and partition data among clients using Dirichlet distribution for non-IID setting."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Loading %s dataset with alpha: %.4f", dataset, alpha)

    # Define data transformations
    if dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            )
        ])
        full_train_dataset = datasets.CIFAR10(
            root='data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(
            root='data', train=False, download=True, transform=transform)
    elif dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_train_dataset = datasets.MNIST(
            root='data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(
            root='data', train=False, download=True, transform=transform)
    elif dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        full_train_dataset = datasets.CelebA(
            root='data', split='train', download=True, transform=transform)
        test_dataset = datasets.CelebA(
            root='data', split='test', download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset.")

    # Get labels
    if hasattr(full_train_dataset, 'targets'):
        labels = np.array(full_train_dataset.targets)
    elif hasattr(full_train_dataset, 'labels'):
        labels = np.array(full_train_dataset.labels)
    else:
        raise ValueError("Dataset does not have labels or targets attribute.")

    # Partition data indices among clients using Dirichlet distribution
    client_indices = partition_data_dirichlet(labels, num_clients, alpha, seed)
    logger.info("Data partitioned among %d clients.", num_clients)

    # Create Subsets for each client
    client_datasets = [
        Subset(full_train_dataset, indices)
        for indices in client_indices
    ]

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0  # Ensure compatibility with multiprocessing
    )

    return client_datasets, test_loader

def partition_data_dirichlet(labels, num_clients, alpha, seed=42):
    """Partition data indices among clients using Dirichlet distribution.

    Args:
        labels (array): Array of labels for the dataset.
        num_clients (int): Number of clients.
        alpha (float): Dirichlet distribution parameter.
        seed (int): Random seed for reproducibility.

    Returns:
        client_indices (list): List of index arrays for each client.
    """
    np.random.seed(seed)
    num_classes = np.max(labels) + 1
    label_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        np.random.shuffle(label_indices[c])
        proportions = np.random.dirichlet(
            alpha * np.ones(num_clients)
        )
        # Balance proportions so that each client gets data
        proportions = np.array([
            p * (len(idx) < len(labels) / num_clients)
            for p, idx in zip(proportions, client_indices)
        ])
        proportions = proportions / proportions.sum()
        splits = (proportions * len(label_indices[c])).astype(int)
        # Adjust splits to ensure all samples are assigned
        splits[-1] = len(label_indices[c]) - splits[:-1].sum()
        idx_list = np.split(label_indices[c], np.cumsum(splits)[:-1])
        for idx, client_idx in zip(idx_list, client_indices):
            client_idx.extend(idx.tolist())

    # Shuffle indices for each client
    for idx in client_indices:
        np.random.shuffle(idx)

    return client_indices

def get_network(model, channel, num_classes, im_size=(32, 32), device='cpu'):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNetBN':
        net = AlexNetBN(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11(channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN':
        net = ResNet18BN(channel=channel, num_classes=num_classes)
    else:
        raise ValueError('Unknown model: %s' % model)

    net = net.to(device)

    return net

def get_default_convnet_setting():
    """
    Provides default settings for the ConvNet architecture.

    Returns:
        tuple: (net_width, net_depth, net_act, net_norm, net_pooling)
    """
    net_width = 128
    net_depth = 3
    net_act = 'relu'
    net_norm = 'instancenorm'
    net_pooling = 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling
