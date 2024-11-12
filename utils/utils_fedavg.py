# utils_fedavg.py

import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import logging
from utils.networks import MLP, ConvNet, LeNet, AlexNet, AlexNetBN, VGG11, VGG11BN, ResNet18, ResNet18BN_AP, ResNet18BN
import copy
import random

# Configure logging
logger = logging.getLogger(__name__)

def randomize_labels(dataset):
    """Randomly switch labels of the dataset."""
    randomized_dataset = copy.deepcopy(dataset)
    if hasattr(randomized_dataset, 'targets'):
        labels = np.array(randomized_dataset.targets)
        np.random.shuffle(labels)
        randomized_dataset.targets = labels.tolist()
    elif hasattr(randomized_dataset, 'labels'):
        labels = np.array(randomized_dataset.labels)
        np.random.shuffle(labels)
        randomized_dataset.labels = labels.tolist()
    else:
        raise AttributeError("Dataset does not have 'targets' or 'labels' attribute.")
    return randomized_dataset

def load_data(dataset, alpha, num_clients):
    """Load and partition data among clients using Dirichlet distribution for non-IID setting."""
    logger.info("Loading %s dataset with alpha: %.4f", dataset, alpha)
    if dataset == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        full_train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    elif dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        full_train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    elif dataset == 'CelebA':
        transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        full_train_dataset = datasets.CelebA(root='data', split='train', download=True, transform=transform)
        test_dataset = datasets.CelebA(root='data', split='test', download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset.")

    # Split data into non-IID partitions
    targets = np.array(full_train_dataset.targets if hasattr(full_train_dataset, 'targets') else full_train_dataset.labels)
    client_indices = partition_data(targets, num_clients, alpha)
    logger.info("Data partitioned among %d clients.", num_clients)

    # Return the full dataset and client indices
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    return full_train_dataset, client_indices, test_loader

def load_client_data(client_id, args):
    """Load the data for a specific client based on client_id."""
    # Load the full training dataset
    if args.dataset == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        full_train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    elif args.dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        full_train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    elif args.dataset == 'CelebA':
        transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        full_train_dataset = datasets.CelebA(root='data', split='train', download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset.")

    # Ensure the same partitioning is applied
    targets = np.array(full_train_dataset.targets if hasattr(full_train_dataset, 'targets') else full_train_dataset.labels)
    client_indices = partition_data(targets, args.num_clients, args.alpha)

    # Get the indices for the specified client
    indices = client_indices[client_id]

    # Create a Subset for the client
    client_dataset = Subset(full_train_dataset, indices)
    return client_dataset

def partition_data(targets, num_clients, alpha):
    """Partition data using Dirichlet distribution."""
    num_classes = np.max(targets) + 1
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        client_splits = np.split(idx_c, proportions)
        for i, split in enumerate(client_splits):
            client_indices[i].extend(split)
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    logger.info("Data partitioning complete.")
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
