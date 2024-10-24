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
from utils.utils import load_data, partition_data_unique_rounds, load_partitions, get_network, plot_accuracies, save_partitions
from torch.utils.data import Subset

def setup_main_logger(log_dir):
    """
    Sets up the main logger to log to both console and file.
    """
    os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists

    logger = logging.getLogger('FedAF.Main')
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if already added
    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'fedaf.log'))
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregation-Free Federated Learning with Client Data Condensation (FedAF)")
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'MNIST', 'CelebA'],
                        help='Dataset to use: MNIST, CIFAR10, CelebA')
    parser.add_argument('--model', type=str, default='ConvNet', choices=['ConvNet', 'ResNet18', 'MLP', 'LeNet', 'AlexNet', 'VGG11'],
                        help='Model architecture: ConvNet, ResNet18, MLP, LeNet, AlexNet, VGG11')
    parser.add_argument('--rounds', type=int, default=20, help='Number of communication rounds')
    parser.add_argument('--ipc', type=int, default=50, help='Instances per class')
    parser.add_argument('--global_steps', type=int, default=500, help='Global training steps')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--alpha', type=float, default=0.1, help='Dirichlet distribution parameter for non-IID data')
    parser.add_argument('--honesty_ratio', type=float, default=1.0, help='Ratio of honest clients (0 to 1)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use: cpu or cuda')
    parser.add_argument('--lr_img', type=float, default=1.0, help='Learning rate for images')
    
    # Static lambda arguments
    parser.add_argument('--lambda_cdc', type=float, default=0.8, help='Lambda for Client Data Condensation')
    parser.add_argument('--lambda_glob', type=float, default=0.8, help='Lambda for Global Aggregation')
    
    # Additional arguments
    parser.add_argument('--init', type=str, default='real', choices=['real', 'random'],
                        help='Initialization method for synthetic data: real, random')
    parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for Softmax Smoothing')
    parser.add_argument('--gamma', type=float, default=0.9, help='Resampling factor for model parameters')
    parser.add_argument('--method', type=str, default='DM', choices=['DM', 'Random'],
                        help='Initialization method for synthetic data: DM, Random')
    parser.add_argument('--Iteration', type=int, default=1000, help='Number of training iterations for synthetic data.')
    parser.add_argument('--eval_it_pool', type=int, nargs='+', default=[500, 1000],
                        help='Iterations at which to visualize synthetic data.')
    parser.add_argument('--save_image_dir', type=str, default='/home/t914a431/images',
                        help='Directory to save synthetic data visualizations.')
    parser.add_argument('--save_path', type=str, default='/home/t914a431/result',
                        help='Directory to save synthetic data.')
    parser.add_argument('--logits_dir', type=str, default='/home/t914a431/logits',
                        help='Directory to save logits.')
    parser.add_argument('--models_dir', type=str, default='/home/t914a431/models',
                        help='Directory to save models.')
    parser.add_argument('--plots_dir', type=str, default='/home/t914a431/plots',
                        help='Directory to save plots.')
    parser.add_argument('--log_dir', type=str, default='/home/t914a431/log/',
                        help='Directory to save logs.')
    parser.add_argument('--partition_dir', type=str, default='/home/t914a431/partitions_per_round',
                        help='Directory where data partitions per round are saved.')

    args = parser.parse_args()
    return args

def initialize_global_model(args, logger):
    """
    Initializes a random global model and saves it so that clients can access it.
    """
    model = get_network(args.model, args.channel, args.num_classes, args.im_size, device=args.device)
    model_dir = os.path.join(args.models_dir, args.dataset, args.model, str(args.num_clients), str(args.honesty_ratio))
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'fedaf_global_model_0.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Global model initialized and saved to {model_path}.")
    time.sleep(2)

def calculate_and_save_logits_worker(args_tuple):
    """
    Worker function for calculating and saving logits.

    Args:
        args_tuple (tuple): Tuple containing (client_id, train_data, args_dict, round_num).

    Returns:
        str or None: Path to the saved logits, or None if an error occurred.
    """
    client_id, train_data, args_dict, round_num = args_tuple
    try:
        # Initialize Client
        client = Client(client_id, train_data, args_dict)
        client.calculate_and_save_logits(round_num)

        # Log when client completes calculating logits
        logger = logging.getLogger('FedAF.Main')
        logger.info(f"Client {client_id} has completed calculating and saving logits for round {round_num}.")

        # Return the round_logit_path
        return client.round_logit_path
    except Exception as e:
        logger = logging.getLogger('FedAF.Main')
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
            logger = logging.getLogger('FedAF.Main')
            logger.info(f"Client {client_id}: No data available or insufficient data for all classes. Skipping condensation.")
            return False  # Indicate that condensation was skipped

        # Proceed with data condensation
        logger = logging.getLogger('FedAF.Main')
        logger.info(f"Client {client_id}: Initializing synthetic data for round {round_num}.")
        client.initialize_synthetic_data(round_num)

        logger.info(f"Client {client_id}: Training synthetic data for round {round_num}.")
        client.train_synthetic_data(round_num)

        logger.info(f"Client {client_id}: Data condensation completed successfully for round {round_num}.")
        return True  # Indicate that condensation succeeded

    except Exception as e:
        logger = logging.getLogger('FedAF.Main')
        logger.exception(f"Client {client_id}: Error during data condensation - {e}")
        return False  # Indicate failure during condensation

def aggregate_logits(logit_dirs: list, num_classes: int, v_r: str, device: str = "cpu") -> list:
    """
    Aggregates class-wise logits from all clients using their logit directories.

    Args:
        logit_dirs (list): List of logit directory paths from clients.
        num_classes (int): Number of classes.
        v_r (str): Indicator for the type of logits ('V' or 'R').
        device (str): Device to use for tensor operations ('cpu' or 'cuda').

    Returns:
        list: Aggregated logits per class.
    """
    aggregated_logits = [torch.zeros(num_classes, device=device) for _ in range(num_classes)]
    class_counts = [0] * num_classes  # To track non-zero logits count

    for logit_dir in logit_dirs:
        if not os.path.exists(logit_dir):
            logger = logging.getLogger('FedAF.Main')
            logger.warning(f"Logit directory {logit_dir} does not exist. Skipping.")
            continue
        try:
            # Iterate over each class's logit file
            for c in range(num_classes):
                logit_file = os.path.join(logit_dir, f'Vkc_{c}.pt') if v_r == 'V' else os.path.join(logit_dir, f'Rkc_{c}.pt')
                if os.path.isfile(logit_file):
                    client_logit = torch.load(logit_file, map_location=device)
                    if client_logit.sum().item() > 0:
                        aggregated_logits[c] += client_logit
                        class_counts[c] += 1
                    else:
                        continue
                else:
                    logger = logging.getLogger('FedAF.Main')
                    logger.warning(f"Logit file {logit_file} does not exist. Skipping.")
        except Exception as e:
            logger = logging.getLogger('FedAF.Main')
            logger.error(f"Error loading logits from directory {logit_dir}: {e}")
            continue

    for c in range(num_classes):
        if class_counts[c] > 0:
            aggregated_logits[c] /= class_counts[c]
            logger = logging.getLogger('FedAF.Main')
            logger.info(f"Aggregated logits for class {c} from {class_counts[c]} clients.")
        else:
            logger = logging.getLogger('FedAF.Main')
            logger.warning(f"No valid logits for class {c} from any client. Initializing aggregated logits with zeros.")
            aggregated_logits[c] = torch.zeros(num_classes, device=device)

    return aggregated_logits

def save_aggregated_logits(aggregated_logits: list, args, round_num: int, v_r: str, logger):
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
        logger.info(f"Server: Aggregated logits saved to {global_logits_path}.")
    except Exception as e:
        logger.error(f"Server: Error saving aggregated logits to {global_logits_path} - {e}")

def simulate():
    args = parse_args()
    logger = setup_main_logger(args.log_dir)
    logger.info("FedAF Main Logger Initialized.")
    
    base_dataset = load_data(args.dataset, data_path='/home/t914a431/data', train=True)
    
    if args.dataset == 'CIFAR10':
        args.channel = 3 
        args.num_classes = 10 
        args.im_size = (32, 32)
    elif args.dataset == 'MNIST':
        args.channel = 1
        args.num_classes = 10
        args.im_size = (28, 28)
    elif args.dataset == 'CelebA':
        args.channel = 3
        args.num_classes = 2
        args.im_size = (64, 64)
    else:
        logger.error(f"Unsupported dataset: {args.dataset}")
        return
    
    if not os.path.exists(args.partition_dir):
        logger.error(f"Partitions directory {args.partition_dir} does not exist. Please generate partitions first.")
    else:
        logger.info(f"Partitions directory {args.partition_dir} found. Loading existing partitions.")
        
    # Load partitions
    client_datasets_per_round = load_partitions(
        dataset=base_dataset,
        num_clients=args.num_clients,
        num_rounds=args.rounds,
        partition_dir=args.partition_dir,
        dataset_name=args.dataset,
        model_name=args.model,
        honesty_ratio=args.honesty_ratio
    )

    # Initialize the global model
    model_dir = os.path.join(args.models_dir, args.dataset, args.model, str(args.num_clients), str(args.honesty_ratio))
    global_model_path = os.path.join(model_dir, 'fedaf_global_model_0.pth')
    if not os.path.exists(global_model_path):
        logger.info("[+] Initializing Global Model")
        initialize_global_model(args, logger)

    test_accuracies = []

    # Main communication rounds
    for r in range(1, args.rounds + 1):
        logger.info(f"---  Round: {r}/{args.rounds}  ---")

        # Load partitioned data for each client for the current round
        current_round = r - 1  # Zero-based indexing
        client_datasets = client_datasets_per_round[current_round]

        # Step 1: Clients calculate and save their class-wise logits
        client_args = [
            (client_id, client_datasets[client_id], vars(args), r)
            for client_id in range(args.num_clients)
        ]

        with multiprocessing.Pool(processes=args.num_clients) as pool:
            logit_paths = pool.map(calculate_and_save_logits_worker, client_args)

        logger.info("Clients have completed calculating and saving logits.")

        # Step 2: Aggregate logits from available clients
        valid_logit_paths = [path for path in logit_paths if path is not None]
        if not valid_logit_paths:
            logger.error(f"No valid logits were calculated by any client in round {r}. Skipping round.")
            logger.info(f"--- Round Ended: {r}/{args.rounds}  ---")
            continue  # Skip the rest of the round if no logits are available

        aggregated_logits = aggregate_logits(valid_logit_paths, args.num_classes, 'V', device=args.device)
        save_aggregated_logits(aggregated_logits, args, r, 'V', logger)

        # Step 3: Clients perform Data Condensation on synthetic data S
        with multiprocessing.Pool(processes=args.num_clients) as pool:
            condensation_status = pool.map(data_condensation_worker, client_args)

        # Step 4: Identify clients that succeeded in data condensation
        successful_clients = [client_id for client_id, status in zip(range(args.num_clients), condensation_status) if status]
        if not successful_clients:
            logger.error(f"No clients succeeded in data condensation in round {r}. Skipping server update.")
            logger.info(f"--- Round Ended: {r}/{args.rounds}  ---")
            continue  # Skip the server update if no clients succeeded

        # Prepare logit paths for successful clients
        successful_logit_paths = [logit_paths[client_id] for client_id in successful_clients]

        # Step 5: Aggregate logits for labels ('R') from successful clients
        aggregated_labels = aggregate_logits(successful_logit_paths, args.num_classes, 'R', device=args.device)
        save_aggregated_logits(aggregated_labels, args, r, 'R', logger)

        # Step 6: Server updates the global model using aggregated soft labels R & synthetic data S
        server_accuracy = server_update(
            model_name=args.model,
            data=args.dataset,
            num_partitions=args.num_clients,
            round_num=r,
            ipc=args.ipc,
            method=args.method,
            hratio=args.honesty_ratio,
            temperature=args.temperature,
            num_epochs=args.global_steps,
            lambda_cdc=args.lambda_cdc,
            lambda_glob=args.lambda_glob,
            device=args.device,
            logger=logger
        )

        test_accuracies.append(server_accuracy)

        logger.info(f"--- Round Ended: {r}/{args.rounds}  ---")

    # Plot and save test accuracy graph after all rounds
    plot_accuracies(
        test_accuracies=test_accuracies,  # Collected accuracies during server_update
        model_name=args.model,
        dataset=args.dataset,
        alpha=args.alpha,
        num_partitions=args.num_clients,
        save_dir=args.plots_dir
    )

    logger.info("Federated Aggregation-Free Learning completed. Test accuracy graph saved.")

if __name__ == '__main__':
    simulate()
