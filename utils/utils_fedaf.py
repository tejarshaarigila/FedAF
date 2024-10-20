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

def compute_swd(logits1, logits2, num_projections=100):
    """
    Computes the Sliced Wasserstein Distance (SWD) between two sets of logits.

    Args:
        logits1 (torch.Tensor or np.ndarray): First set of logits.
        logits2 (torch.Tensor or np.ndarray): Second set of logits.
        num_projections (int): Number of random projections to use.

    Returns:
        float: Computed SWD value.
    """
    try:
        if isinstance(logits1, torch.Tensor):
            logits1 = logits1.detach().cpu().numpy()
        if isinstance(logits2, torch.Tensor):
            logits2 = logits2.detach().cpu().numpy()

        # Check if logits are valid arrays
        if logits1.ndim == 0 or logits2.ndim == 0:
            logger.warning("compute_swd: One of the logits is zero-dimensional. Returning 0.0.")
            return 0.0  # Return zero distance if logits are scalars or zero-dimensional

        dimensions = logits1.shape[0]
        if dimensions == 0:
            logger.warning("compute_swd: Logits have zero dimensions. Returning 0.0.")
            return 0.0  # Return zero distance if dimension is zero

        # Generate random projections
        projections = np.random.normal(0, 1, (num_projections, dimensions))
        projections /= np.linalg.norm(projections, axis=1, keepdims=True)

        # Project the logits
        projected_logits1 = np.dot(projections, logits1)
        projected_logits2 = np.dot(projections, logits2)

        # Compute Wasserstein distance for each projection
        distances = wasserstein_distance(projected_logits1, projected_logits2, axis=0)
        average_distance = np.mean(distances)

        logger.debug(f"compute_swd: SWD computed with average distance {average_distance:.4f}.")
        return average_distance
    except Exception as e:
        logger.error(f"compute_swd: Error computing SWD - {e}")
        return 0.0  # Return zero in case of error to avoid crashing

def calculate_logits_labels(model_net, partition, num_classes, device, path, ipc, temperature):
    """
    Calculates and saves class-wise averaged logits (V(k,c)) and probabilities (R(k,c)).

    Args:
        model_net (torch.nn.Module): The global model.
        partition (torch.utils.data.Subset): Client's data partition.
        num_classes (int): Number of classes.
        device (str): Device to perform computations on.
        path (str): Directory path to save logits.
        ipc (int): Instances per class.
        temperature (float): Temperature parameter for softmax.
    """
    try:
        # Create subdirectories if they don't exist
        os.makedirs(path, exist_ok=True)

        # Create DataLoader for the client's partition
        dataloader = DataLoader(
            partition,
            batch_size=256,
            shuffle=False,
            num_workers=0  # Set to 0 for multiprocessing compatibility
        )

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
                        logger.warning(f"calculate_logits_labels: Label {label} out of range [0, {num_classes-1}]. Skipping.")

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
    except Exception as e:
        logger.error(f"calculate_logits_labels: Error during logits and labels calculation - {e}")

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
            model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) 
                          if f.endswith('.pth') and f.startswith('fedaf')]
            if model_files:
                latest_model_file = max(model_files, key=os.path.getmtime)
                net = get_network(model_name, channel, num_classes, im_size, device=device)
                state_dict = torch.load(latest_model_file, map_location=device)
                net.load_state_dict(state_dict)
                logger.info(f"load_latest_model: Loaded model from {latest_model_file}")
                return net
            else:
                logger.warning(f"load_latest_model: No model files found in {model_dir}. Initializing a new model.")
        else:
            logger.warning(f"load_latest_model: Model directory {model_dir} is empty or does not exist. Initializing a new model.")

        # If no model exists, initialize a new one
        net = get_network(model_name, channel, num_classes, im_size, device=device)
        return net
    except Exception as e:
        logger.error(f"load_latest_model: Error loading the latest model - {e}")
        # Initialize a new model in case of error
        net = get_network(model_name, channel, num_classes, im_size, device=device)
        return net

def get_dataset(dataset, data_path, num_partitions, alpha, seed=42):
    """
    Loads and partitions the dataset using a Dirichlet distribution for non-IID data.

    Args:
        dataset (str): Dataset name ('MNIST' or 'CIFAR10').
        data_path (str): Path to download/load the dataset.
        num_partitions (int): Number of client partitions.
        alpha (float): Dirichlet distribution parameter controlling data heterogeneity.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (channel, im_size, num_classes, class_names, mean, std, list of Subset datasets for training, test dataset, test DataLoader)
    """
    try:
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)

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
            logger.error(f"get_dataset: Unknown dataset: {dataset}")
            raise ValueError(f"Unknown dataset: {dataset}")

        labels = np.array(base_train.targets) if hasattr(base_train, 'targets') else np.array(base_train.labels)
        indices = [[] for _ in range(num_partitions)]

        for c in range(num_classes):
            class_indices = np.where(labels == c)[0]
            np.random.shuffle(class_indices)
            proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_partitions))
            proportions = np.array(proportions)
            proportions = proportions / proportions.sum()
            class_splits = (proportions * len(class_indices)).astype(int)

            # Adjust to ensure all samples are assigned
            if class_splits.sum() < len(class_indices):
                class_splits[-1] += len(class_indices) - class_splits.sum()
            elif class_splits.sum() > len(class_indices):
                class_splits[-1] -= class_splits.sum() - len(class_indices)

            split_indices = np.split(class_indices, np.cumsum(class_splits)[:-1]) if num_partitions > 1 else [class_indices]
            for idx, client_idx in zip(split_indices, indices):
                client_idx.extend(idx.tolist())

        # Create subsets for each partition
        dst_train_partitions = [Subset(base_train, idx) for idx in indices]

        # Create test DataLoader
        testloader = DataLoader(
            dst_test,
            batch_size=256,
            shuffle=False,
            num_workers=0  # Set to 0 for multiprocessing compatibility
        )

        logger.info(f"get_dataset: Loaded and partitioned {dataset} dataset into {num_partitions} partitions with alpha={alpha}.")
        return channel, im_size, num_classes, class_names, mean, std, dst_train_partitions, dst_test, testloader
    except Exception as e:
        logger.error(f"get_dataset: Error loading and partitioning dataset - {e}")
        raise e

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

def get_network(model, channel, num_classes, im_size=(32, 32), device='cpu'):
    """
    Initializes and returns the specified network architecture.

    Args:
        model (str): Name of the model architecture.
        channel (int): Number of input channels.
        num_classes (int): Number of output classes.
        im_size (tuple): Image size (height, width).
        device (str): Device to load the model on.

    Returns:
        torch.nn.Module: Initialized network.
    """
    try:
        torch.random.manual_seed(int(time.time() * 1000) % 100000)
        net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

        if model == 'MLP':
            net = MLP(channel=channel, num_classes=num_classes)
        elif model == 'ConvNet':
            net = ConvNet(
                channel=channel,
                num_classes=num_classes,
                net_width=net_width,
                net_depth=net_depth,
                net_act=net_act,
                net_norm=net_norm,
                net_pooling=net_pooling,
                im_size=im_size
            )
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

        elif model.startswith('ConvNetD'):
            depth = int(model[-1])
            net = ConvNet(
                channel=channel,
                num_classes=num_classes,
                net_width=net_width,
                net_depth=depth,
                net_act=net_act,
                net_norm=net_norm,
                net_pooling=net_pooling,
                im_size=im_size
            )
        elif model.startswith('ConvNetW'):
            width = int(model[-2:])
            net = ConvNet(
                channel=channel,
                num_classes=num_classes,
                net_width=width,
                net_depth=net_depth,
                net_act=net_act,
                net_norm=net_norm,
                net_pooling=net_pooling,
                im_size=im_size
            )
        elif model.startswith('ConvNetA'):
            act_type = model[len('ConvNetA'):]
            norm = net_norm
            pooling = net_pooling
            net = ConvNet(
                channel=channel,
                num_classes=num_classes,
                net_width=net_width,
                net_depth=net_depth,
                net_act=act_type.lower(),
                net_norm=norm,
                net_pooling=pooling,
                im_size=im_size
            )
        elif model.startswith('ConvNetN'):
            norm_type = model[len('ConvNetN'):]
            pooling = net_pooling
            net = ConvNet(
                channel=channel,
                num_classes=num_classes,
                net_width=net_width,
                net_depth=net_depth,
                net_act=net_act,
                net_norm=norm_type.lower(),
                net_pooling=pooling,
                im_size=im_size
            )
        elif model.startswith('ConvNetP'):
            pooling_type = model[len('ConvNetP'):]
            net = ConvNet(
                channel=channel,
                num_classes=num_classes,
                net_width=net_width,
                net_depth=net_depth,
                net_act=net_act,
                net_norm=net_norm,
                net_pooling=pooling_type.lower(),
                im_size=im_size
            )
        else:
            logger.error(f"get_network: Unknown model: {model}")
            raise ValueError(f"Unknown model: {model}")

        net = net.to(device)
        logger.info(f"get_network: Initialized model {model} with {channel} channels, {num_classes} classes, image size {im_size}.")
        return net
    except Exception as e:
        logger.error(f"get_network: Error initializing model {model} - {e}")
        raise e

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
    try:
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
        logger.info(f"get_eval_pool: Evaluation pool prepared with models: {model_eval_pool}")
        return model_eval_pool
    except Exception as e:
        logger.error(f"get_eval_pool: Error preparing evaluation pool - {e}")
        return [model_eval]  # Fallback to the default evaluation model

def aggregate_logits(logit_paths, num_classes, v_r):
    """
    Aggregates class-wise logits from all clients using their logit paths.

    Args:
        logit_paths (list): List of logit paths from clients.
        num_classes (int): Number of classes.
        v_r (str): Indicator for the type of logits ('V' or 'R').

    Returns:
        list of torch.Tensor: Aggregated logits per class.
    """
    aggregated_logits = [torch.zeros(num_classes) for _ in range(num_classes)]
    count = [0 for _ in range(num_classes)]

    for client_logit_path in logit_paths:
        if client_logit_path is None:
            continue
        for c in range(num_classes):
            client_Vkc_path = os.path.join(client_logit_path, f'{v_r}kc_{c}.pt')
            if os.path.exists(client_Vkc_path):
                try:
                    client_logit = torch.load(client_Vkc_path, map_location='cpu')
                    if torch.all(client_logit == 0):
                        logger.warning(f"aggregate_logits: Client logit at {client_Vkc_path} is all zeros. Skipping.")
                        continue
                    aggregated_logits[c] += client_logit
                    count[c] += 1
                except Exception as e:
                    logger.error(f"aggregate_logits: Error loading logits from {client_Vkc_path} - {e}")
            else:
                logger.warning(f"aggregate_logits: Logit file {client_Vkc_path} does not exist. Skipping.")

    # Average the logits
    for c in range(num_classes):
        if count[c] > 0:
            aggregated_logits[c] /= count[c]
            logger.debug(f"aggregate_logits: Class {c} aggregated logits averaged over {count[c]} clients.")
        else:
            aggregated_logits[c] = torch.zeros(num_classes)
            logger.warning(f"aggregate_logits: No logits found for class {c}. Initialized aggregated logits to zeros.")

    logger.info("aggregate_logits: Aggregated logits computed successfully.")
    return aggregated_logits

def save_aggregated_logits(aggregated_logits, args, r, v_r):
    """
    Saves the aggregated logits to a global directory accessible by all clients.

    Args:
        aggregated_logits (list of torch.Tensor): Aggregated logits per class.
        args (ARGS): Configuration parameters.
        r (int): Current round number.
        v_r (str): Indicator for the type of logits ('V' or 'R').
    """
    try:
        logits_dir = os.path.join(args.logits_dir, 'Global')
        os.makedirs(logits_dir, exist_ok=True)
        global_logits_path = os.path.join(logits_dir, f'Round{r}_Global_{v_r}c.pt')
        torch.save(aggregated_logits, global_logits_path)
        logger.info(f"save_aggregated_logits: Aggregated logits saved to {global_logits_path}.")
    except Exception as e:
        logger.error(f"save_aggregated_logits: Error saving aggregated logits - {e}")
