# utils_fedavg.py

import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import logging
from config import DATASET_REGISTRY, TORCHVISION_DATASET
from utils.networks import MLP, ConvNet, LeNet, AlexNet, AlexNetBN, VGG11, VGG11BN, ResNet18, ResNet18BN_AP, ResNet18BN
import copy
import random

# Configure logging
logger = logging.getLogger(__name__)


def _build_transform(dataset_name):
    """Normalisation transform from the single dataset registry (shared with FedAF)."""
    meta = DATASET_REGISTRY[dataset_name]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=meta['mean'], std=meta['std']),
    ])


def _load_torchvision(dataset_name, data_path, train):
    """Instantiate a registry dataset (MNIST/FMNIST/CIFAR10/CIFAR100), normalised."""
    if dataset_name not in TORCHVISION_DATASET:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                         f"Known: {sorted(TORCHVISION_DATASET)}")
    ds_cls = getattr(datasets, TORCHVISION_DATASET[dataset_name])
    return ds_cls(root=data_path, train=train, download=True,
                  transform=_build_transform(dataset_name))

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

def load_data(cfg):
    """Load and Dirichlet-partition the configured dataset among clients.

    Uses the shared registry transform so FedAvg, FedAF and the plotter all
    normalise identically (and therefore evaluate on the same footing).
    """
    logger.info("Loading %s dataset with alpha: %.4f", cfg.dataset, cfg.alpha)
    full_train_dataset = _load_torchvision(cfg.dataset, cfg.data_path, train=True)
    test_dataset = _load_torchvision(cfg.dataset, cfg.data_path, train=False)

    targets = np.array(full_train_dataset.targets
                       if hasattr(full_train_dataset, 'targets') else full_train_dataset.labels)

    # Split data using Dirichlet distribution (seeded for cross-process consistency)
    client_indices = partition_data(targets, cfg.num_clients, cfg.alpha, seed=cfg.seed)
    logger.info("Data partitioned among %d clients.", cfg.num_clients)

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    return full_train_dataset, client_indices, test_loader

def load_client_data(client_id, cfg):
    """Return the Subset of training data assigned to `client_id`."""
    full_train_dataset = _load_torchvision(cfg.dataset, cfg.data_path, train=True)

    targets = np.array(full_train_dataset.targets
                       if hasattr(full_train_dataset, 'targets') else full_train_dataset.labels)

    client_indices = partition_data(targets, cfg.num_clients, cfg.alpha, seed=cfg.seed)
    indices = client_indices[client_id]

    return Subset(full_train_dataset, indices)

def partition_data(targets, num_clients, alpha, seed=None):
    """Partition data using a Dirichlet distribution.

    `seed` makes the split reproducible AND identical across processes: the
    server (load_data) and each worker (load_client_data) must derive the same
    partition, which is not guaranteed by inherited RNG state under the 'spawn'
    start method (the default on macOS/Windows).
    """
    if seed is not None:
        np.random.seed(seed)
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
