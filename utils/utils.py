# utils/utils.py

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from scipy.stats import wasserstein_distance
import logging
import time
from networks import (
    MLP, ConvNet, LeNet, AlexNet, AlexNetBN, VGG11, VGG11BN,
    ResNet18, ResNet18BN_AP, ResNet18BN
)

# Configure logging for utilities
logger = logging.getLogger('FedAF.Utils')
if not logger.handlers:
    log_directory = "/home/t914a431/log"
    os.makedirs(log_directory, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_directory, 'utils.log'))
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)


def load_data(dataset_name, data_path='data'):
    """Load the dataset and apply necessary transformations."""
    logger.info("Loading dataset: %s", dataset_name)
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2023, 0.1994, 0.2010))
        ])
        dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    elif dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    elif dataset_name == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        dataset = datasets.CelebA(root=data_path, split='train', download=True, transform=transform)
    else:
        logger.error(f"Unsupported dataset: {dataset_name}")
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    logger.info("Dataset loaded successfully.")
    return dataset

def partition_data_per_round(dataset, num_clients, num_rounds, alpha, seed=42):
    """
    Partition dataset into unique, non-overlapping subsets for each client and each round,
    considering the total number of rounds and allowing for missing data for some clients.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to partition.
        num_clients (int): Number of clients.
        num_rounds (int): Number of communication rounds.
        alpha (float): Dirichlet distribution parameter for data heterogeneity.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary with keys as round numbers and values as lists of client indices.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Starting dataset partitioning per round with Dirichlet distribution.")

    # Extract labels and create indices for each class
    labels = np.array(dataset.targets)
    num_classes = np.max(labels) + 1
    label_indices = [np.where(labels == i)[0].tolist() for i in range(num_classes)]

    # Shuffle the indices for each class
    for c in range(num_classes):
        np.random.shuffle(label_indices[c])

    client_indices_per_round = {round_num: [[] for _ in range(num_clients)] for round_num in range(num_rounds)}

    for round_num in range(num_rounds):
        logger.info(f"Partitioning data for Round {round_num + 1}/{num_rounds}")
        for c in range(num_classes):
            available_indices = label_indices[c]

            if len(available_indices) == 0:
                logger.warning(f"No more data for class {c} in round {round_num + 1}. Skipping this class for this round.")
                continue

            # Distribute class data using Dirichlet distribution across clients
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = proportions / proportions.sum()  # Normalize proportions to ensure sum is 1

            # Calculate the number of samples for each client based on Dirichlet proportions
            samples_per_client = (proportions * len(available_indices)).astype(int)

            # Adjust to ensure total samples match available data for the class
            samples_per_client[-1] = len(available_indices) - samples_per_client[:-1].sum()

            # Split indices for this class based on the calculated sample proportions
            split_indices = np.split(available_indices, np.cumsum(samples_per_client)[:-1])

            for client_id, indices in enumerate(split_indices):
                if len(indices) == 0:
                    logger.info(f"Client {client_id} received no data for class {c} in round {round_num + 1}.")
                client_indices_per_round[round_num][client_id].extend(indices.tolist())

    logger.info("Dataset partitioning completed successfully.")
    return client_indices_per_round

def save_partitions(client_indices_per_round, save_dir='partitions_per_round'):
    """Save partitions for each client for each round."""
    os.makedirs(save_dir, exist_ok=True)
    for round_num, client_lists in client_indices_per_round.items():
        round_dir = os.path.join(save_dir, f'round_{round_num}')
        os.makedirs(round_dir, exist_ok=True)
        for client_id, indices in enumerate(client_lists):
            partition_path = os.path.join(round_dir, f'client_{client_id}_partition.pkl')
            with open(partition_path, 'wb') as f:
                pickle.dump(indices, f)
    logger.info(f"All data partitions saved in directory: {save_dir}")


def load_partitions(dataset, num_clients, num_rounds, partition_dir='partitions_per_round'):
    """
    Load pre-partitioned data for each client for each round.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset.
        num_clients (int): Number of clients.
        num_rounds (int): Number of communication rounds.
        partition_dir (str): Directory where partitions are saved.

    Returns:
        dict: A dictionary with keys as round numbers and values as lists of Subset datasets for each client.
              If a partition is missing, the client will have no data for that round.
    """
    client_datasets_per_round = {}
    for round_num in range(num_rounds):
        round_dir = os.path.join(partition_dir, f'round_{round_num}')
        client_datasets = []
        for client_id in range(num_clients):
            partition_path = os.path.join(round_dir, f'client_{client_id}_partition.pkl')
            
            # Check if the partition file exists
            if os.path.exists(partition_path):
                # Load the partition if it exists
                with open(partition_path, 'rb') as f:
                    indices = pickle.load(f)
                client_subset = Subset(dataset, indices)
                client_datasets.append(client_subset)
            else:
                # If partition is missing, assume the client has no data for this round
                logger.warning(f"Partition missing for Client {client_id} in Round {round_num}. Skipping this client.")
                client_datasets.append(Subset(dataset, []))  # Empty subset for this client

        client_datasets_per_round[round_num] = client_datasets

    logger.info("All available data partitions loaded successfully.")
    return client_datasets_per_round

def get_network(model_name, channel, num_classes, im_size=(32, 32), device='cpu'):
    """Initializes the network based on the model name."""
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model_name == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model_name == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                     net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model_name == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model_name == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model_name == 'AlexNetBN':
        net = AlexNetBN(channel=channel, num_classes=num_classes)
    elif model_name == 'VGG11':
        net = VGG11(channel=channel, num_classes=num_classes)
    elif model_name == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model_name == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model_name == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model_name == 'ResNet18BN':
        net = ResNet18BN(channel=channel, num_classes=num_classes)
    else:
        logger.error(f"Unsupported model '{model_name}'.")
        raise ValueError(f"Unsupported model '{model_name}'.")
    
    net = net.to(device)
    logger.info(f"Initialized model '{model_name}' on device '{device}'.")
    return net


def get_default_convnet_setting():
    """Provides default settings for the ConvNet architecture."""
    net_width = 128
    net_depth = 3
    net_act = 'relu'
    net_norm = 'instancenorm'
    net_pooling = 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling


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
                state_dict = torch.load(latest_model_file, map_location=device)
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


def plot_accuracies(test_accuracies, model_name, dataset, alpha, num_partitions, num_clients=None, save_dir='plots'):
    """
    Plots and saves the test accuracies over communication rounds.

    Args:
        test_accuracies (list): List of test accuracies per round.
        model_name (str): Name of the model used.
        dataset (str): Name of the dataset used.
        alpha (float): Dirichlet distribution parameter.
        num_partitions (int): Number of client partitions.
        num_clients (int, optional): Number of clients (FedAvg only).
        save_dir (str): Directory to save the plot.
    """
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    rounds = range(1, len(test_accuracies) + 1)
    plt.figure()
    plt.plot(rounds, test_accuracies, marker='o')
    plt.title(f"Test Accuracy over Rounds\nModel: {model_name}, Dataset: {dataset}, Alpha: {alpha}")
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    if num_clients:
        plt.legend([f"{num_clients} Clients"])
    save_path = os.path.join(save_dir, f"accuracy_{model_name}_{dataset}_alpha{alpha}.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"plot_accuracies: Test accuracy graph saved to {save_path}.")
