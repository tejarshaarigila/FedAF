# utils/utils.py

import logging
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from utils.networks import (
    MLP, ConvNet, LeNet, AlexNet, AlexNetBN, VGG11,
    VGG11BN, ResNet18, ResNet18BN_AP, ResNet18BN
)

# Configure logger
logger = logging.getLogger(__name__)

def ensure_directory_exists(path: str):
    """
    Ensures that the specified directory exists. If it does not exist, creates it.
    
    Args:
        path (str): The directory path to check/create.
    """
    try:
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Directory ensured: {path}")
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise


def partition_data_unique_rounds(dataset, num_clients, num_rounds, alpha, seed=42, logger=None):
    """
    Partition the dataset for each round using Dirichlet distribution.

    Args:
        dataset (Dataset): The dataset to partition.
        num_clients (int): Number of clients.
        num_rounds (int): Number of communication rounds.
        alpha (float): Dirichlet distribution parameter for data heterogeneity.
        seed (int): Random seed for reproducibility.

    Returns:
        list: A list where each element corresponds to a round and contains a list of lists of client indices.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if logger is None:
        logger = logging.getLogger('GeneratePartitions')
    
    logger.info("Starting dataset partitioning for each round with Dirichlet distribution.")

    client_indices_per_round = []
    labels = np.array(dataset.targets)
    num_classes = np.max(labels) + 1

    for round_num in range(num_rounds):
        # Extract labels and create indices for each class
        label_indices = [np.where(labels == i)[0].tolist() for i in range(num_classes)]

        # Shuffle the indices for each class
        for c in range(num_classes):
            np.random.shuffle(label_indices[c])

        client_indices = [[] for _ in range(num_clients)]
        
        # Allocate data to clients using Dirichlet distribution
        for c in range(num_classes):
            available_indices = label_indices[c]
            if len(available_indices) == 0:
                continue  # Skip if no data is available for this class
            
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (np.array(proportions) * len(available_indices)).astype(int)
            
            # Ensure no negative or zero-sized allocation by adjusting proportions
            proportions[-1] = len(available_indices) - np.sum(proportions[:-1])
            if proportions[-1] < 0:
                proportions[-1] = 0  # Adjust in case rounding issues occur

            start = 0
            for client_id, num_samples in enumerate(proportions):
                end = start + num_samples
                client_indices[client_id].extend(available_indices[start:end])
                start = end
        
        client_indices_per_round.append(client_indices)

    logger.info("Dataset partitioning for each round completed successfully.")
    return client_indices_per_round
    

def save_partitions(client_indices_per_round, save_dir):
    """
    Save the client indices for each round and each client.

    Args:
        client_indices_per_round (list): A list where each element corresponds to a round and contains a list of lists of client indices.
        save_dir (str): Directory where partitions are to be saved.
    """
    for round_num, client_indices in enumerate(client_indices_per_round):
        round_dir = os.path.join(save_dir, f'round_{round_num}')
        os.makedirs(round_dir, exist_ok=True)
        for client_id, indices in enumerate(client_indices):
            partition_path = os.path.join(round_dir, f'client_{client_id}_partition.pkl')
            try:
                with open(partition_path, 'wb') as f:
                    pickle.dump(indices, f)
                logger.info(f"Saved partition for Client {client_id} in Round {round_num} at {partition_path}")
            except Exception as e:
                logger.error(f"Failed to save partition for Client {client_id} in Round {round_num}: {e}")


def load_partitions(dataset, num_clients, num_rounds, partition_dir, dataset_name, model_name, honesty_ratio):
    """
    Load pre-partitioned data for each client for each round.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset.
        num_clients (int): Number of clients.
        num_rounds (int): Number of communication rounds.
        partition_dir (str): Directory where partitions are saved.
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model.
        honesty_ratio (float): Honesty ratio.
    """
    client_datasets_per_round = {}

    for round_num in range(num_rounds):
        round_dir = os.path.join(partition_dir, f'round_{round_num}')
        client_datasets = []
        logger.info(f"--- Round {round_num} ---")
        for client_id in range(num_clients):
            partition_path = os.path.join(round_dir, f'client_{client_id}_partition.pkl')
            logger.info(f"Checking partition path: {partition_path}")
            if os.path.exists(partition_path):
                try:
                    with open(partition_path, 'rb') as f:
                        indices = pickle.load(f)
                    num_images = len(indices)
                    logger.info(f"Client {client_id}: Assigned {num_images} images.")

                    # Calculate the number of images per class
                    labels = np.array(dataset.targets)[indices]
                    unique_classes, class_counts = np.unique(labels, return_counts=True)
                    class_distribution = dict(zip(unique_classes, class_counts))
                    logger.info(f"Client {client_id}: Images per class: {class_distribution}")

                    client_subset = Subset(dataset, indices)
                    client_datasets.append(client_subset)
                except Exception as e:
                    logger.error(f"Error loading partition for Client {client_id} in Round {round_num}: {e}")
                    client_datasets.append(Subset(dataset, []))
            else:
                logger.warning(f"Round {round_num}, Client {client_id}: Partition file missing. Assigning empty dataset.")
                client_datasets.append(Subset(dataset, []))
        client_datasets_per_round[round_num] = client_datasets

    logger.info("All data partitions loaded successfully.")
    return client_datasets_per_round

def get_network(model_name, channel, num_classes, im_size=(32, 32), device='cpu'):
    """Initializes the network based on the model name."""
    torch.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model_name == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model_name == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    else:
        logger.error(f"Unsupported model '{model_name}'.")
        raise ValueError(f"Unsupported model '{model_name}'.")

    net = net.to(device)
    logger.info(f"Initialized model '{model_name}' on device '{device}'.")
    return net


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

    if logits1.ndim == 0 or logits2.ndim == 0:
        return 0.0

    dimensions = logits1.shape[0]
    if dimensions == 0:
        return 0.0

    projections = np.random.normal(0, 1, (num_projections, dimensions))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    projected_logits1 = projections.dot(logits1)
    projected_logits2 = projections.dot(logits2)

    distances = [wasserstein_distance([projected_logits1[i]], [projected_logits2[i]]) for i in range(num_projections)]
    return np.mean(distances)

def calculate_logits_labels(model_net, partition, num_classes, device, path, ipc, temperature, logit_type='V'):
    """
    Calculates and saves class-wise averaged logits (Vkc or Rkc).

    Args:
        model_net (torch.nn.Module): The global model.
        partition (Subset): Client's data partition.
        num_classes (int): Number of classes.
        device (torch.device): Device to perform computations on.
        path (str): Directory path to save logits.
        ipc (int): Instances per class.
        temperature (float): Temperature parameter for softmax.
        logit_type (str): Type of logits to calculate ('V' or 'R').

    Raises:
        ValueError: If `logit_type` is neither 'V' nor 'R'.
    """
    if logit_type not in ['V', 'R']:
        logger.error(f"Invalid logit_type '{logit_type}'. Must be 'V' or 'R'.")
        raise ValueError(f"Invalid logit_type '{logit_type}'. Must be 'V' or 'R'.")

    os.makedirs(path, exist_ok=True)
    dataloader = DataLoader(partition, batch_size=256, shuffle=False)

    logits_by_class = [[] for _ in range(num_classes)]

    model_net.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model_net(images)
            if logit_type == 'R':
                probs = F.softmax(logits / temperature, dim=1)
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    if 0 <= label < num_classes:
                        logits_by_class[label].append(probs[i].unsqueeze(0))
                    else:
                        logger.warning(f"Label {label} is out of range. Skipping.")
            elif logit_type == 'V':
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    if 0 <= label < num_classes:
                        logits_by_class[label].append(logits[i].unsqueeze(0))
                    else:
                        logger.warning(f"Label {label} is out of range. Skipping.")

    logits_avg = []
    for c in range(num_classes):
        if len(logits_by_class[c]) >= ipc:
            class_logits = torch.cat(logits_by_class[c], dim=0)
            avg_logit = class_logits.mean(dim=0)
        else:
            avg_logit = torch.zeros(num_classes, device=device)
            logger.warning(f"Not enough instances for class {c}. Initialized with zeros.")
        logits_avg.append(avg_logit)

    # Save the averaged logits
    try:
        for c in range(num_classes):
            torch.save(logits_avg[c], os.path.join(path, f'Vkc_{c}.pt'))
        logger.info(f"Saved averaged logits for class {c} to {path}.")
    except Exception as e:
        logger.error(f"Error saving {logit_type}kc logits: {e}")

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
                state_dict = torch.load(latest_model_file, map_location=device)
                model.load_state_dict(state_dict)
                logger.info(f"Loaded model from {latest_model_file}.")
                return model
        # If no model exists, initialize a new one
        logger.info("Model directory is empty or no valid model found. Initializing a new model.")
        model = get_network(
            model_name=model_name,
            channel=channel,
            num_classes=num_classes,
            im_size=im_size,
            device=device
        )
        return model
    except Exception as e:
        logger.error(f"Error loading the latest model: {e}")
        # Initialize a new model in case of error
        model = get_network(
            model_name=model_name,
            channel=channel,
            num_classes=num_classes,
            im_size=im_size,
            device=device
        )
        return model

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
            t_c = F.softmax(avg_logit / temperature, dim=0)
            t_list.append(t_c)
        else:
            # Initialize with uniform distribution if no data for class c
            t_list.append(torch.ones(num_classes, device=device) / num_classes)
            logger.warning(f"No synthetic data for class {c}. Initialized T with uniform distribution.")

    t_tensor = torch.stack(t_list)  # [num_classes, num_classes]
    logger.info("Computed class-wise averaged soft labels T.")
    return t_tensor


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
            logger.warning('Replacing BatchNorm with InstanceNorm in evaluation.')
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


def plot_accuracies(test_accuracies, model_name, dataset_name, alpha, num_clients, save_dir='plots'):
    """
    Plots and saves the test accuracies over communication rounds.

    Args:
        test_accuracies (list): List of test accuracies per round.
        model_name (str): Name of the model used.
        dataset_name (str): Name of the dataset used.
        alpha (float): Dirichlet distribution parameter.
        num_clients (int): Number of clients.
        save_dir (str): Directory to save the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    rounds = range(1, len(test_accuracies) + 1)
    plt.figure()
    plt.plot(rounds, test_accuracies, marker='o')
    plt.title(f"Test Accuracy over Rounds\nModel: {model_name}, Dataset: {dataset_name}, Alpha: {alpha}")
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.legend([f"{num_clients} Clients"])
    save_path = os.path.join(save_dir, f"accuracy_{model_name}_{dataset_name}_alpha{alpha}.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Test accuracy graph saved to {save_path}.")
