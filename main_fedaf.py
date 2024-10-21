# main_fedaf.py

import os
import torch
import numpy as np
import logging
import argparse
import multiprocessing
from client.client_fedaf import Client
from server.server_fedaf import server_update
from utils.utils_fedaf import load_data, get_network
from multiprocessing import Pool

# Ensure the log directory exists
log_dir = "/home/t914a431/log/"
os.makedirs(log_dir, exist_ok=True)

# Configure logging to save log file in the specified directory
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=os.path.join(log_dir, 'fedaf.log'),  # Log file path
    filemode='w'
)
logger = logging.getLogger(__name__)


def get_dataset_config(dataset_name: str) -> dict:
    """
    Returns the configuration for the specified dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'CIFAR10', 'MNIST').

    Returns:
        dict: Configuration parameters for the dataset.
    """
    dataset_configs = {
        'CIFAR10': {
            'im_size': (32, 32),
            'channel': 3,
            'num_classes': 10,
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2023, 0.1994, 0.2010]
        },
        'MNIST': {
            'im_size': (28, 28),
            'channel': 1,
            'num_classes': 10,
            'mean': [0.1307],
            'std': [0.3081]
        },
        # PENDING
    }

    config = dataset_configs.get(dataset_name.upper())
    if config is None:
        logger.error(f"Unsupported dataset: {dataset_name}. Supported datasets are: {list(dataset_configs.keys())}")
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets are: {list(dataset_configs.keys())}")
    return config


def get_args():
    parser = argparse.ArgumentParser(description="Aggregaton Free Federated Learning with Client Data Condensation")
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to use: MNIST, CIFAR10')
    parser.add_argument('--model', type=str, default='ConvNet', help='Model architecture: ConvNet, ResNet')
    parser.add_argument('--rounds', type=int, default=50, help='Number of communication rounds')
    parser.add_argument('--ipc', type=int, default=50, help='Instances per class')
    parser.add_argument('--global_steps', type=int, default=500, help='Global training steps')
    parser.add_argument('--num_partitions', type=int, default=10, help='Number of client partitions')
    parser.add_argument('--alpha', type=float, default=0.1, help='Dirichlet distribution parameter for non-IID data')
    parser.add_argument('--honesty_ratio', type=float, default=1.0, help='Ratio of honest clients')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: cpu or cuda')
    parser.add_argument('--lr_img', type=float, default=1.0, help='Learning rate for images')
    
    # Added arguments
    parser.add_argument('--temperature', type=float, default=2, help='Temperature for Softmax Smoothing')
    parser.add_argument('--gamma', type=float, default=0.9, help='Resampling factor for model parameters')
    parser.add_argument('--method', type=str, default='DM', help='Initialization method for synthetic data: DM, Random')
    parser.add_argument('--Iteration', type=int, default=500, help='Number of training iterations for synthetic data.')
    parser.add_argument('--eval_it_pool', type=int, nargs='+', default=[0, 100, 200, 300, 400, 500], help='Iterations at which to evaluate and visualize synthetic data.')
    parser.add_argument('--save_image_dir', type=str, default='/home/t914a431/images', help='Directory to save synthetic data visualizations.')
    parser.add_argument('--save_path', type=str, default='/home/t914a431/result', help='Directory to save synthetic data.')
    parser.add_argument('--logits_dir', type=str, default='/home/t914a431/logits', help='Directory to save logits.')

    args = parser.parse_args()
    return args


def initialize_global_model(args):
    """
    Initializes a random global model and saves it so that clients can access it.
    """
    model = get_network(args.model, args.channel, args.num_classes, args.im_size, device=args.device)
    model_dir = f'./models/{args.dataset}/{args.model}/{args.num_partitions}/{args.honesty_ratio}'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'fedaf_global_model_0.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Global model initialized and saved to {model_path}.")


def simulate():
    args = get_args()
    rounds = args.rounds

    # Get dataset-specific configurations
    dataset_config = get_dataset_config(args.dataset)
    args.im_size = dataset_config['im_size']
    args.channel = dataset_config['channel']
    args.num_classes = dataset_config['num_classes']
    args.mean = dataset_config['mean']
    args.std = dataset_config['std']

    # Create necessary directories using updated configurations
    model_dir = f'./models/{args.dataset}/{args.model}/{args.num_partitions}/{args.honesty_ratio}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.logits_dir, exist_ok=True)
    os.makedirs(args.save_image_dir, exist_ok=True)

    # Load and partition the dataset
    client_datasets, test_loader = load_data(
        dataset=args.dataset,
        alpha=args.alpha,
        num_clients=args.num_partitions,
        seed=42  # For reproducibility
    )

    # Initialize the global model and save it
    global_model_path = os.path.join(model_dir, 'fedaf_global_model_0.pth')
    if not os.path.exists(global_model_path):
        logger.info("[+] Initializing Global Model")
        initialize_global_model(args)

    args_dict = vars(args)  # Convert args Namespace to a dictionary

    # Main communication rounds
    for r in range(1, rounds + 1):
        logger.info(f"---  Round: {r}/{rounds}  ---")

        # Step 1: Clients calculate and save their class-wise logits
        client_args = [
            (client_id, train_data, args_dict, r)
            for client_id, train_data in enumerate(client_datasets)
        ]
        with multiprocessing.Pool(processes=args.num_partitions) as pool:
            logit_paths = pool.map(calculate_and_save_logits_worker, client_args)

        # Step 2: Server aggregates logits and saves aggregated logits for clients
        aggregated_logits = aggregate_logits(logit_paths, args.num_classes, 'V')
        save_aggregated_logits(aggregated_logits, args, r, 'V')

        # Step 3: Clients perform Data Condensation on synthetic data S
        with multiprocessing.Pool(processes=args.num_partitions) as pool:
            pool.map(data_condensation_worker, client_args)

        # Step 4: Server updates the global model using aggregated soft labels R & synthetic data S
        aggregated_labels = aggregate_logits(logit_paths, args.num_classes, 'R')
        save_aggregated_logits(aggregated_labels, args, r, 'R')
        server_update(
            model_name=args.model,
            data=args.dataset,
            num_partitions=args.num_partitions,
            round_num=r,
            ipc=args.ipc,
            method=args.method,
            hratio=args.honesty_ratio,
            temperature=args.temperature,
            num_epochs=args.global_steps,
            device=args.device
        )
        logger.info(f"--- Round Ended: {r}/{rounds}  ---")


def calculate_and_save_logits_worker(args_tuple):
    """
    Worker function for calculating and saving logits.

    Args:
        args_tuple (tuple): Tuple containing (client_id, train_data, args_dict, round_num).

    Returns:
        str or None: Path to the saved logits, or None if an error occurred.
    """
    client_id, train_data, args_dict, r = args_tuple
    try:
        # Initialize Client
        client = Client(client_id, train_data, args_dict)
        client.calculate_and_save_logits(r)
        # Log when client completes calculating logits
        logger.info(f"Client {client_id} has completed calculating and saving logits for round {r}.")
        return client.logit_path
    except Exception as e:
        logger.exception(f"Exception in client {client_id} during logits calculation: {e}")
        return None


def data_condensation_worker(args_tuple):
    """
    Worker function for data condensation.

    Args:
        args_tuple (tuple): Tuple containing (client_id, train_data, args_dict, round_num).
    """
    client_id, train_data, args_dict, r = args_tuple
    try:
        # Initialize Client
        client = Client(client_id, train_data, args_dict)
        client.initialize_synthetic_data(r)
        client.train_synthetic_data(r)
        # Log when client completes data condensation
        logger.info(f"Client {client_id} has completed data condensation for round {r}.")
    except Exception as e:
        logger.exception(f"Exception in client {client_id} during data condensation: {e}")


def aggregate_logits(logit_paths: list, num_classes: int, v_r: str) -> list:
    """
    Aggregates class-wise logits from all clients using their logit paths.

    Args:
        logit_paths (list): List of logit paths from clients.
        num_classes (int): Number of classes.
        v_r (str): Indicator for the type of logits ('V' or 'R').

    Returns:
        list: Aggregated logits per class.
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
                    if not torch.all(client_logit == 0):
                        aggregated_logits[c] += client_logit
                        count[c] += 1
                except Exception as e:
                    logger.error(f"Server: Error loading logits for class {c} from {client_Vkc_path} - {e}")
            else:
                logger.warning(f"Server: Missing logits for class {c} in {client_logit_path}. Skipping.")

    # Average the logits
    for c in range(num_classes):
        if count[c] > 0:
            aggregated_logits[c] /= count[c]
        else:
            aggregated_logits[c] = torch.zeros(num_classes)  # Default if no clients have data for class c

    logger.info("Server: Aggregated logits computed.")
    return aggregated_logits


def save_aggregated_logits(aggregated_logits: list, args, round_num: int, v_r: str):
    """
    Saves the aggregated logits to a global directory accessible by all clients.

    Args:
        aggregated_logits (list): Aggregated logits per class.
        args (Namespace): Parsed arguments.
        round_num (int): Current round number.
        v_r (str): Indicator for the type of logits ('V' or 'R').
    """
    logits_dir = os.path.join(args.logits_dir, 'Global')
    os.makedirs(logits_dir, exist_ok=True)
    global_logits_path = os.path.join(logits_dir, f'Round{round_num}_Global_{v_r}c.pt')
    torch.save(aggregated_logits, global_logits_path)
    logger.info(f"Server: Aggregated logits saved to {global_logits_path}.")


if __name__ == '__main__':
    simulate()
