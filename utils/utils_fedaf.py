# utils_fedaf.py

import time
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from scipy.stats import wasserstein_distance
import logging
from utils.networks import MLP, ConvNet, LeNet, AlexNet, AlexNetBN, VGG11, VGG11BN, ResNet18, ResNet18BN_AP, ResNet18BN

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

def compute_swd(logits1, logits2, num_projections=100):
    """
    Computes the Sliced Wasserstein Distance (SWD) between two sets of logits.
    """
    if isinstance(logits1, torch.Tensor):
        logits1 = logits1.detach().cpu().numpy()
    if isinstance(logits2, torch.Tensor):
        logits2 = logits2.detach().cpu().numpy()

    # Check if logits are valid arrays
    if logits1.ndim == 0 or logits2.ndim == 0:
        return 0.0  # Return zero distance if logits are scalars or zero-dimensional

    dimensions = logits1.shape[0]
    if dimensions == 0:
        return 0.0  # Return zero distance if dimension is zero

    # Generate projections
    projections = np.random.normal(0, 1, (num_projections, dimensions))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    # Project the logits
    projected_logits1 = projections.dot(logits1)
    projected_logits2 = projections.dot(logits2)

    # Compute Wasserstein distance for each projection
    distances = []
    for i in range(num_projections):
        distance = wasserstein_distance([projected_logits1[i]], [projected_logits2[i]])
        distances.append(distance)
    average_distance = np.mean(distances)
    return average_distance

def calculate_logits_labels(model_net, partition, num_classes, device, path, ipc, temperature):
    """
    Calculates and saves class-wise averaged logits (V(k,c)) and probabilities (R(k,c)).

    Args:
        model_net (torch.nn.Module): The global model.
        partition (torch.utils.data.Dataset): Client's data partition.
        num_classes (int): Number of classes.
        device (torch.device): Device to perform computations on.
        path (str): Directory path to save logits.
        ipc (int): Instances per class.
        temperature (float): Temperature parameter for softmax.
    """

    # Create subdirectories if they don't exist
    os.makedirs(path, exist_ok=True)

    # Create DataLoader for the client's partition
    dataloader = DataLoader(partition, batch_size=256, shuffle=False)

    # Initialize storage for logits and probabilities
    logits_by_class = [torch.empty((0, num_classes), device=device) for _ in range(num_classes)]
    probs_by_class = [torch.empty((0, num_classes), device=device) for _ in range(num_classes)]

    model_net.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model_net(images)
            probs = F.softmax(logits / temperature, dim=1)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                logits_by_class[label] = torch.cat((logits_by_class[label], logits[i].unsqueeze(0)), dim=0)
                probs_by_class[label] = torch.cat((probs_by_class[label], probs[i].unsqueeze(0)), dim=0)

    # Average logits and probabilities per class
    logits_avg = []
    probs_avg = []
    for c in range(num_classes):
        if logits_by_class[c].size(0) >= ipc:
            avg_logit = logits_by_class[c].mean(dim=0)
            avg_prob = probs_by_class[c].mean(dim=0)
        else:
            avg_logit = torch.zeros(num_classes, device=device)
            avg_prob = torch.zeros(num_classes, device=device)
        logits_avg.append(avg_logit)
        probs_avg.append(avg_prob)

    # Save the averaged logits and probabilities
    try:
        for c in range(num_classes):
            if logits_by_class[c].size(0) >= ipc:
                avg_logit = logits_by_class[c].mean(dim=0) 
                avg_prob = probs_by_class[c].mean(dim=0)
                logger.debug(f"calculate_logits_labels: Class {c} - Avg Logit: {avg_logit}, Avg Prob: {avg_prob}")
            else:
                avg_logit = torch.zeros(num_classes, device=device)
                avg_prob = torch.zeros(num_classes, device=device)
                logger.warning(f"calculate_logits_labels: Not enough instances for class {c}. Initialized avg_logit and avg_prob with zeros.")
            logits_avg.append(avg_logit)
            probs_avg.append(avg_prob)

        # Save the averaged logits and probabilities
        try:
            for c in range(num_classes):
                torch.save(logits_avg[c], os.path.join(path, f'Vkc_{c}.pt'))
                torch.save(probs_avg[c], os.path.join(path, f'Rkc_{c}.pt'))
            logger.info(f"calculate_logits_labels: Saved averaged logits and probabilities to {path}.")
        except Exception as e:
            logger.error(f"calculate_logits_labels: Error saving logits and probabilities - {e}")
    except Exception as e:
        logger.error(f"Error saving logits and probabilities: {e}")

def load_latest_model(model_dir, model_name, channel, num_classes, im_size, device):
    """
    Loads the latest global model from the model directory.

    Args:
        model_dir (str): Directory containing model checkpoints.
        model_name (str): Name of the model architecture.
        channel (int): Number of input channels.
        num_classes (int): Number of output classes.
        im_size (tuple): Image size (height, width).
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded model.
    """
    try:
        if os.path.exists(model_dir) and os.listdir(model_dir):
            model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) 
                          if f.endswith('.pth') and f.startswith('fedaf')]
            if model_files:
                latest_model_file = max(model_files, key=os.path.getmtime)
                net = get_network(model_name, channel, num_classes, im_size, device=device)
                state_dict = torch.load(latest_model_file, map_location=device, weights_only=True)
                net.load_state_dict(state_dict)
                logger.info(f"Loaded model from {latest_model_file}")
                return net
        # If no model exists, initialize a new one
        logger.info("Model directory is empty or no valid model found. Initializing a new model.")
        net = get_network(model_name, channel, num_classes, im_size).to(device)
        return net
    except Exception as e:
        logger.error(f"Error loading the latest model: {e}")
        # Initialize a new model in case of error
        net = get_network(model_name, channel, num_classes, im_size).to(device)
        return net

def get_dataset(dataset, data_path, num_partitions, alpha):
    """
    Loads and partitions the dataset using a Dirichlet distribution for non-IID data.

    Args:
        dataset (str): Dataset name ('MNIST' or 'CIFAR10').
        data_path (str): Path to download/load the dataset.
        num_partitions (int): Number of client partitions.
        alpha (float): Dirichlet distribution parameter controlling data heterogeneity.

    Returns:
        tuple: (channel, im_size, num_classes, class_names, mean, std, list of Subset datasets for training, test dataset, test DataLoader)
    """
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        base_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]
    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        base_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = base_train.classes
    else:
        logger.error(f"Unknown dataset: {dataset}")
        raise ValueError(f"Unknown dataset: {dataset}")
    
    labels = np.array(base_train.targets)
    indices = [[] for _ in range(num_partitions)]
    
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet(np.repeat(alpha, num_partitions))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        class_splits = np.split(class_indices, proportions)
        for idx in range(num_partitions):
            if idx < len(class_splits):
                indices[idx].extend(class_splits[idx])
    
    # Create subsets for each partition
    dst_train_partitions = [Subset(base_train, idx) for idx in indices]
    
    # Create test DataLoader
    testloader = DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=4)
    
    return channel, im_size, num_classes, class_names, mean, std, dst_train_partitions, dst_test, testloader

def get_network(model, channel, num_classes, im_size=(32, 32)):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNetBN':
        net = AlexNetBN(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11( channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN':
        net = ResNet18BN(channel=channel, num_classes=num_classes)

    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='relu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetASwish':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='swish', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetASwishBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='swish', net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='none', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='layernorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='none', im_size=im_size)
    elif model == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='maxpooling', im_size=im_size)
    elif model == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='avgpooling', im_size=im_size)

    else:
        net = None
        exit('unknown model: %s'%model)

    gpu_num = torch.cuda.device_count()
    if gpu_num>0:
        device = 'cuda'
        if gpu_num>1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'
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

def get_eval_pool(eval_mode, model, model_eval):
    """
    Prepares a pool of models for evaluation based on the evaluation mode.

    Args:
        eval_mode (str): Evaluation mode ('S', 'SS', etc.).
        model (str): Current model architecture.
        model_eval (str): Model architecture for evaluation.

    Returns:
        list: List containing model architectures for evaluation.
    """
    if eval_mode == 'S':  # Self
        if 'BN' in model:
            logger.warning('Attention: Replacing BatchNorm with InstanceNorm in evaluation.')
        try:
            bn_index = model.index('BN')
            model_eval_pool = [model[:bn_index]]
        except ValueError:
            model_eval_pool = [model]
    elif eval_mode == 'SS':  # Self-Self
        model_eval_pool = [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool

