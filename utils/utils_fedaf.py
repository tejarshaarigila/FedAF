# utils/utils_fedaf.py

import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from scipy.stats import wasserstein_distance
import logging
from utils.networks import (
    MLP, ConvNet, LeNet, AlexNet, AlexNetBN, VGG11,
    VGG11BN, ResNet18, ResNet18BN_AP, ResNet18BN
)

# Configure logging for utilities
logger = logging.getLogger('FedAF.Utils')
if not logger.handlers:
    file_handler = logging.FileHandler("/home/t914a431/log/utils_fedaf.log")
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)


def load_data(dataset, alpha, num_clients, seed=42):
    """
    Load and partition data among clients using Dirichlet distribution for non-IID setting.

    Args:
        dataset (str): Dataset name ('CIFAR10', 'MNIST', etc.).
        alpha (float): Dirichlet distribution parameter for data partitioning.
        num_clients (int): Number of client partitions.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (list of Subset datasets for clients, DataLoader for test dataset)
    """
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
            root='/home/t914a431/data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(
            root='/home/t914a431/data', train=False, download=True, transform=transform)
    elif dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_train_dataset = datasets.MNIST(
            root='/home/t914a431/data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(
            root='/home/t914a431/data', train=False, download=True, transform=transform)
    elif dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        full_train_dataset = datasets.CelebA(
            root='/home/t914a431/data', split='train', download=True, transform=transform)
        test_dataset = datasets.CelebA(
            root='/home/t914a431/data', split='test', download=True, transform=transform)
    else:
        logger.error(f"Unsupported dataset: {dataset}")
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Get labels
    if hasattr(full_train_dataset, 'targets'):
        labels = np.array(full_train_dataset.targets)
    elif hasattr(full_train_dataset, 'labels'):
        labels = np.array(full_train_dataset.labels)
    else:
        logger.error("Dataset does not have labels or targets attribute.")
        raise ValueError("Dataset does not have labels or targets attribute.")

    # Partition data indices among clients using Dirichlet distribution
    client_indices = partition_data_dirichlet(
        labels=labels,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed
    )
    logger.info("Data partitioned among %d clients.", num_clients)

    # Create Subsets for each client
    client_datasets = [Subset(full_train_dataset, indices) for indices in client_indices]

    # Create test DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4  # Adjust based on your system
    )
    logger.info("Test DataLoader created.")

    return client_datasets, test_loader


def partition_data_dirichlet(labels, num_clients, alpha, seed=42):
    """
    Partition data indices among clients using Dirichlet distribution.

    Args:
        labels (np.ndarray): Array of labels for the dataset.
        num_clients (int): Number of clients.
        alpha (float): Dirichlet distribution parameter.
        seed (int): Random seed for reproducibility.

    Returns:
        list: List of index lists for each client.
    """
    np.random.seed(seed)
    num_classes = np.max(labels) + 1
    label_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        np.random.shuffle(label_indices[c])
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        proportions = np.array([
            p * (len(idx) < len(labels) / num_clients)
            for p, idx in zip(proportions, client_indices)
        ])
        if proportions.sum() == 0:
            proportions = np.ones(num_clients) / num_clients
        else:
            proportions = proportions / proportions.sum()
        splits = (proportions * len(label_indices[c])).astype(int)
        splits[-1] = len(label_indices[c]) - splits[:-1].sum()
        idx_list = np.split(label_indices[c], np.cumsum(splits)[:-1])
        for idx, client_idx in zip(idx_list, client_indices):
            client_idx.extend(idx.tolist())

    # Shuffle indices for each client
    for idx in client_indices:
        np.random.shuffle(idx)

    logger.info("Data partitioning with Dirichlet distribution completed.")
    return client_indices


def compute_swd(logits1, logits2, num_projections=100):
    """
    Computes the Sliced Wasserstein Distance (SWD) between two sets of logits.

    Args:
        logits1 (torch.Tensor or np.ndarray): First set of logits.
        logits2 (torch.Tensor or np.ndarray): Second set of logits.
        num_projections (int): Number of random projections.

    Returns:
        float: Average SWD over all projections.
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
        partition (Subset): Client's data partition.
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
                if 0 <= label < num_classes:
                    logits_by_class[label] = torch.cat((logits_by_class[label], logits[i].unsqueeze(0)), dim=0)
                    probs_by_class[label] = torch.cat((probs_by_class[label], probs[i].unsqueeze(0)), dim=0)
                else:
                    logger.warning(f"calculate_logits_labels: Label {label} is out of range. Skipping.")

    # Average logits and probabilities per class
    logits_avg = []
    probs_avg = []
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


def load_latest_model(model_dir, model_name, channel, num_classes, im_size, device):
    """
    Loads the latest global model from the model directory.

    Args:
        model_dir (str): Directory containing model checkpoints.
        model_name (str): Name of the model architecture.
        channel (int): Number of input channels.
        num_classes (int): Number of output classes.
        im_size (tuple): Image size (height, width).
        device (str): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded model.
    """
    try:
        if os.path.exists(model_dir) and os.listdir(model_dir):
            model_files = [
                os.path.join(model_dir, f) for f in os.listdir(model_dir)
                if f.endswith('.pth') and f.startswith('fedaf')
            ]
            if model_files:
                latest_model_file = max(model_files, key=os.path.getmtime)
                model = get_network(
                    model_name=model_name,
                    channel=channel,
                    num_classes=num_classes,
                    im_size=im_size,
                    device=device
                )
                state_dict = torch.load(latest_model_file, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
                logger.info(f"load_latest_model: Loaded model from {latest_model_file}.")
                return model
        # If no model exists, initialize a new one
        logger.info("load_latest_model: Model directory is empty or no valid model found. Initializing a new model.")
        model = get_network(
            model_name=model_name,
            channel=channel,
            num_classes=num_classes,
            im_size=im_size,
            device=device
        )
        return model
    except Exception as e:
        logger.error(f"load_latest_model: Error loading the latest model: {e}")
        # Initialize a new model in case of error
        model = get_network(
            model_name=model_name,
            channel=channel,
            num_classes=num_classes,
            im_size=im_size,
            device=device
        )
        return model


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
        proportions = proportions / proportions.sum()
        splits = (proportions * len(class_indices)).astype(int)
        splits[-1] = len(class_indices) - splits[:-1].sum()
        class_splits = np.split(class_indices, np.cumsum(splits)[:-1])
        for idx, client_idx in zip(class_splits, indices):
            client_idx.extend(idx.tolist())

    # Create subsets for each partition
    dst_train_partitions = [Subset(base_train, idx) for idx in indices]

    # Create test DataLoader
    testloader = DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=4)

    return channel, im_size, num_classes, class_names, mean, std, dst_train_partitions, dst_test, testloader


def get_network(model_name, channel, num_classes, im_size=(32, 32), device='cpu'):
    """
    Initializes the network based on the model name.

    Args:
        model_name (str): Name of the model architecture ('ConvNet', 'ResNet', etc.).
        channel (int): Number of input channels.
        num_classes (int): Number of output classes.
        im_size (tuple): Image size (height, width).
        device (str): Device to use ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: Initialized model.
    """
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model_name == 'MLP':
        model = MLP(channel=channel, num_classes=num_classes)
    elif model_name == 'ConvNet':
        model = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size
        )
    elif model_name == 'LeNet':
        model = LeNet(channel=channel, num_classes=num_classes)
    elif model_name == 'AlexNet':
        model = AlexNet(channel=channel, num_classes=num_classes)
    elif model_name == 'AlexNetBN':
        model = AlexNetBN(channel=channel, num_classes=num_classes)
    elif model_name == 'VGG11':
        model = VGG11(channel=channel, num_classes=num_classes)
    elif model_name == 'VGG11BN':
        model = VGG11BN(channel=channel, num_classes=num_classes)
    elif model_name == 'ResNet18':
        model = ResNet18(channel=channel, num_classes=num_classes)
    elif model_name == 'ResNet18BN_AP':
        model = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model_name == 'ResNet18BN':
        model = ResNet18BN(channel=channel, num_classes=num_classes)
    else:
        logger.error(f"Unsupported model '{model_name}'.")
        raise ValueError(f"Unsupported model '{model_name}'.")

    model = model.to(device)
    logger.info(f"Initialized model '{model_name}' on device '{device}'.")
    return model


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
            logger.warning('get_eval_pool: Replacing BatchNorm with InstanceNorm in evaluation.')
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


def compute_T(model, synthetic_dataset, num_classes, temperature, device):
    """
    Computes the class-wise averaged soft labels T from the model's predictions on the synthetic data.

    Args:
        model (torch.nn.Module): The global model.
        synthetic_dataset (TensorDataset): Synthetic data dataset.
        num_classes (int): Number of classes.
        temperature (float): Temperature parameter for softmax scaling.
        device (str): Device to perform computations on.

    Returns:
        torch.Tensor: Tensor of class-wise averaged soft labels T.
    """
    model.eval()
    class_logits_sum = [torch.zeros(num_classes, device=device) for _ in range(num_classes)]
    class_counts = [0 for _ in range(num_classes)]

    synthetic_loader = DataLoader(synthetic_dataset, batch_size=256, shuffle=False)

    with torch.no_grad():
        for inputs, labels in synthetic_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # [batch_size, num_classes]
            for i in range(inputs.size(0)):
                label = labels[i].item()
                if 0 <= label < num_classes:
                    class_logits_sum[label] += outputs[i]
                    class_counts[label] += 1

    t_list = []
    for c in range(num_classes):
        if class_counts[c] > 0:
            avg_logit = class_logits_sum[c] / class_counts[c]
            t_c = nn.functional.softmax(avg_logit / temperature, dim=0)
            t_list.append(t_c)
        else:
            # Initialize with uniform distribution if no data for class c
            t_list.append(torch.ones(num_classes, device=device) / num_classes)
            logger.warning(f"compute_T: No synthetic data for class {c}. Initialized T with uniform distribution.")

    t_tensor = torch.stack(t_list)  # [num_classes, num_classes]
    logger.info("compute_T: Computed class-wise averaged soft labels T.")
    return t_tensor


def save_aggregated_logits(aggregated_logits, args, round_num, v_r, logger):
    """
    Saves the aggregated logits to a global directory accessible by all clients.

    Args:
        aggregated_logits (list): Aggregated logits per class.
        args (Namespace): Parsed arguments.
        round_num (int): Current round number.
        v_r (str): Indicator for the type of logits ('V' or 'R').
        logger (logging.Logger): Logger instance.
    """
    logits_dir = os.path.join(args.logits_dir, 'Global')
    os.makedirs(logits_dir, exist_ok=True)
    global_logits_path = os.path.join(logits_dir, f'Round{round_num}_Global_{v_r}c.pt')
    try:
        torch.save(aggregated_logits, global_logits_path)
        logger.info(f"save_aggregated_logits: Aggregated logits saved to {global_logits_path}.")
    except Exception as e:
        logger.error(f"save_aggregated_logits: Error saving aggregated logits to {global_logits_path} - {e}")
