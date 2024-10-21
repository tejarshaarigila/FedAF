# main_fedaf.py

import os
import time
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
        # Add more datasets if needed
    }

    config = dataset_configs.get(dataset_name.upper())
    if config is None:
        logger.error(f"Unsupported dataset: {dataset_name}. Supported datasets are: {list(dataset_configs.keys())}")
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets are: {list(dataset_configs.keys())}")
    return config


def get_args():
    parser = argparse.ArgumentParser(description="Aggregation Free Federated Learning with Client Data Condensation")
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to use: MNIST, CIFAR10')
    parser.add_argument('--model', type=str, default='ConvNet', help='Model architecture: ConvNet, ResNet')
    parser.add_argument('--rounds', type=int, default=50, help='Number of communication rounds')
    parser.add_argument('--ipc', type=int, default=50, help='Instances per class')
    parser.add_argument('--global_steps', type=int, default=500, help='Global training steps')
    parser.add_argument('--num_partitions', type=int, default=10, help='Number of client partitions')
    parser.add_argument('--alpha', type=float, default=0.1, help='Dirichlet distribution parameter for non-IID data')
    parser.add_argument('--honesty_ratio', type=int, default=1, help='Ratio of honest clients')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: cpu or cuda')
    parser.add_argument('--lr_img', type=float, default=1.0, help='Learning rate for images')
    
    # Added arguments
    parser.add_argument('--init', type=str, default='real', help='Initialization method for synthetic data: real, random')
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
    model_dir = f'/home/t914a431/models/{args.dataset}/{args.model}/{args.num_partitions}/{args.honesty_ratio}'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'fedaf_global_model_0.pth')
    torch.save(model.state_dict(), model_path)
    time.sleep(2)
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
    model_dir = f'/home/t914a431/models/{args.dataset}/{args.model}/{args.num_partitions}/{args.honesty_ratio}'
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
            (client_id, client_datasets[client_id], args_dict, r)
            for client_id in range(len(client_datasets))
        ]
        
        with multiprocessing.Pool(processes=args.num_partitions) as pool:
            logit_paths = pool.map(calculate_and_save_logits_worker, client_args)

        # Step 2: Aggregate logits from available clients
        valid_logit_paths = [path for path in logit_paths if path is not None]
        if not valid_logit_paths:
            logger.error(f"No valid logits were calculated by any client in round {r}. Skipping round.")
            logger.info(f"--- Round Ended: {r}/{rounds}  ---")
            continue  # Skip the rest of the round if no logits are available

        aggregated_logits = aggregate_logits(valid_logit_paths, args.num_classes, 'V', device=args.device)
        save_aggregated_logits(aggregated_logits, args, r, 'V')

        # Step 3: Clients perform Data Condensation on synthetic data S
        with multiprocessing.Pool(processes=args.num_partitions) as pool:
            condensation_status = pool.map(data_condensation_worker, client_args)

        # Step 4: Identify clients that succeeded in data condensation
        successful_clients = [client_id for client_id, status in zip(range(len(client_datasets)), condensation_status) if status]
        if not successful_clients:
            logger.error(f"No clients succeeded in data condensation in round {r}. Skipping server update.")
            logger.info(f"--- Round Ended: {r}/{rounds}  ---")
            continue  # Skip the server update if no clients succeeded

        # Prepare logit paths for successful clients
        successful_logit_paths = [logit_paths[client_id] for client_id in successful_clients]

        # Step 5: Aggregate logits for labels ('R') from successful clients
        aggregated_labels = aggregate_logits(successful_logit_paths, args.num_classes, 'R', device=args.device)
        save_aggregated_logits(aggregated_labels, args, r, 'R')

        # Step 6: Server updates the global model using aggregated soft labels R & synthetic data S
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

    Returns:
        bool: True if data condensation succeeds, False otherwise.
    """
    client_id, train_data, args_dict, round_num = args_tuple
    try:
        # Initialize Client
        client = Client(client_id, train_data, args_dict)

        # Check if the client should be skipped
        if client.has_no_data():
            logger.info(f"Client {client_id}: No data available or insufficient data for all classes. Skipping condensation.")
            return False  # Indicate that condensation was skipped

        # Proceed with data condensation
        logger.info(f"Client {client_id}: Initializing synthetic data for round {round_num}.")
        client.initialize_synthetic_data(round_num)

        logger.info(f"Client {client_id}: Training synthetic data for round {round_num}.")
        client.train_synthetic_data(round_num)

        logger.info(f"Client {client_id}: Data condensation completed successfully for round {round_num}.")
        return True  # Indicate that condensation succeeded

    except Exception as e:
        logger.exception(f"Client {client_id}: Error during data condensation - {e}")
        return False  # Indicate failure during condensation


def aggregate_logits(logit_paths: list, num_classes: int, v_r: str, device: str = "cpu") -> list:
    """
    Aggregates class-wise logits from all clients using their logit paths.

    Args:
        logit_paths (list): List of logit file paths from clients.
        num_classes (int): Number of classes.
        v_r (str): Indicator for the type of logits ('V' or 'R').
        device (str): Device to use for tensor operations ('cpu' or 'cuda').

    Returns:
        list: Aggregated logits per class.
    """
    aggregated_logits = [torch.zeros(num_classes, device=device) for _ in range(num_classes)]
    class_counts = [0] * num_classes  # To track non-zero logits count

    for path in logit_paths:
        if not os.path.exists(path):
            logger.warning(f"Logit file {path} does not exist. Skipping.")
            continue
        try:
            client_logits = torch.load(path, map_location=device)  # Expecting a list of tensors
            if not isinstance(client_logits, list) or len(client_logits) != num_classes:
                logger.warning(f"Logit file {path} is not in the expected format. Skipping.")
                continue

            for c in range(num_classes):
                # Check if the logit for class c is non-zero
                if client_logits[c].sum().item() > 0:
                    aggregated_logits[c] += client_logits[c]
                    class_counts[c] += 1
                else:
                    # Skip adding if logits for class c are zero
                    continue
        except Exception as e:
            logger.error(f"Error loading logit file {path}: {e}")
            continue

    for c in range(num_classes):
        if class_counts[c] > 0:
            aggregated_logits[c] /= class_counts[c]
            logger.info(f"Aggregated logits for class {c} from {class_counts[c]} clients.")
        else:
            logger.warning(f"No valid logits for class {c} from any client. Initializing aggregated logits with zeros.")
            # Optionally, you can keep it as zeros or handle it differently

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
    try:
        torch.save(aggregated_logits, global_logits_path)
        logger.info(f"Server: Aggregated logits saved to {global_logits_path}.")
    except Exception as e:
        logger.error(f"Server: Error saving aggregated logits to {global_logits_path} - {e}")


if __name__ == '__main__':
    simulate()
