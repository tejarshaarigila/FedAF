# utils_fedaf.py

import time
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

def calculate_logits_labels(model_net, partition, num_classes, device, path, ipc, temperature, logits_type='V'):
    """
    Calculates and saves class-wise averaged logits (V(k,c)) or probabilities (R(k,c)).

    Args:
        model_net (torch.nn.Module): The global model.
        partition (torch.utils.data.Dataset or DataLoader): Data to use for calculations.
        num_classes (int): Number of classes.
        device (torch.device): Device to perform computations on.
        path (str): Directory path to save logits.
        ipc (int): Instances per class.
        temperature (float): Temperature parameter for softmax.
        logits_type (str): 'V' for Vkc logits, 'R' for Rkc logits.
    """

    # Create subdirectories if they don't exist
    os.makedirs(path, exist_ok=True)

    # If partition is a DataLoader, use it directly; otherwise, create a DataLoader
    if isinstance(partition, DataLoader):
        dataloader = partition
    else:
        dataloader = DataLoader(partition, batch_size=256, shuffle=False)

    # Initialize storage for logits
    logits_by_class = [torch.empty((0, num_classes), device=device) for _ in range(num_classes)]

    model_net.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model_net(images)
            if logits_type == 'V':
                output = logits
            elif logits_type == 'R':
                output = F.softmax(logits / temperature, dim=1)
            else:
                raise ValueError("logits_type must be 'V' or 'R'")

            for i in range(labels.size(0)):
                label = labels[i].item()
                logits_by_class[label] = torch.cat((logits_by_class[label], output[i].unsqueeze(0)), dim=0)

    # Average logits per class
    logits_avg = []
    for c in range(num_classes):
        if logits_by_class[c].size(0) >= ipc:
            avg_logit = logits_by_class[c].mean(dim=0)
        else:
            avg_logit = torch.zeros(num_classes, device=device)
        logits_avg.append(avg_logit)

    # Save the averaged logits
    try:
        for c in range(num_classes):
            torch.save(logits_avg[c], os.path.join(path, f'{logits_type}kc_{c}.pt'))
    except Exception as e:
        logger.error(f"Error saving {logits_type}kc logits: {e}")

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
            model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.pth') and f.startswith('fedaf')]
            latest_model_file = max(model_files, key=os.path.getmtime, default=None)

            if latest_model_file:
                net = get_network(model_name, channel, num_classes, im_size).to(device)
                state_dict = torch.load(latest_model_file, map_location=device)
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

def get_base_dataset(dataset_name, data_path, train=True):
    """
    Returns the base dataset without any partitioning, used for reconstructing datasets in subprocesses.
    """
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            # Add other necessary transformations if any
        ])
        dataset = datasets.CIFAR10(data_path, train=train, download=True, transform=transform)
    elif dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
            # Add other necessary transformations if any
        ])
        dataset = datasets.MNIST(data_path, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return dataset

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
    else:
        net = None
        exit('unknown model: %s'%model)

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

def ensure_directory_exists(path):
    """
    Ensures that the directory exists; if not, creates it.

    Args:
        path (str): Directory path to check and create.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def save_aggregated_logits(aggregated_logits, args, r, v_r):
    """
    Saves the aggregated logits to a global directory accessible by all clients.

    Args:
        aggregated_logits (torch.Tensor): Aggregated logits Rc of shape [num_classes,].
        args (ARGS): Argument parser containing configurations.
        r (int): Current round number.
        v_r (str): Type of logits ('V' for pre-condensation, 'R' for post-condensation).
    """
    try:
        logits_dir = os.path.join(args.logits_dir, 'Global')
        os.makedirs(logits_dir, exist_ok=True)
        global_logits_path = os.path.join(logits_dir, f'Round{r}_Global_{v_r}c.pt')
        torch.save(aggregated_logits, global_logits_path)  # Saving single tensor
        logger.info(f"Aggregated {v_r} logits saved to {global_logits_path}.")
    except Exception as e:
        logger.error(f"Error saving aggregated logits: {e}")
