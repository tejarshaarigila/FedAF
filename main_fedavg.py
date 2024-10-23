# main_fedavg.py

import os
import argparse
import torch
from server.server_fedavg import Server
from client.client_fedavg import Client
from utils.utils import (
    load_data,
    load_partitions,
    get_network,
    plot_accuracies
)
import logging
import random
from torch.utils.data import DataLoader
import multiprocessing

def setup_main_logger(log_dir):
    """
    Sets up the main logger to log to both console and file.
    """
    logger = logging.getLogger('FedAvg.Main')
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(os.path.join(log_dir, 'fedavg.log'))
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning with FedAvg")
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'CelebA'],
                        help='Dataset to use (MNIST, CIFAR10, CelebA)')
    parser.add_argument('--model', type=str, default='ConvNet', choices=['ConvNet', 'ResNet'],
                        help='Model to use (ConvNet, ResNet)')
    parser.add_argument('--num_clients', type=int, default=4, help='Number of clients')
    parser.add_argument('--alpha', type=float, default=0.1, help='Dirichlet distribution parameter')
    parser.add_argument('--local_epochs', type=int, default=10, help='Number of local epochs per client')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_rounds', type=int, default=20, help='Number of communication rounds')
    parser.add_argument('--honesty_ratio', type=float, default=1.0, help='Ratio of honest clients (0 to 1)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--partition_dir', type=str, default='/home/t914a431/partitions_per_round',
                        help='Directory where data partitions per round are saved')
    parser.add_argument('--save_dir', type=str, default='/home/t914a431/models', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='/home/t914a431/log/', help='Directory to save logs')
    return parser.parse_args()

def train_client(client_id, train_data, global_model_state, args, honest):
    """Helper function to train a client, skipping the client if no data."""
    if len(train_data) == 0:  # Check if the dataset is empty
        logging.info(f"Client {client_id}: No data for this round. Skipping.")
        return None  # Skip this client if no data
    
    # Recreate the client object with simpler arguments
    client = Client(client_id=client_id, train_data=train_data, args=args)
    client.set_model(global_model_state)
    if not honest:
        train_data = randomize_labels(train_data)
    return client.train()

def set_dataset_params(args):
    """Set dataset-specific parameters for the given dataset."""
    if args.dataset == 'MNIST':
        args.channel = 1
        args.num_classes = 10
        args.im_size = (28, 28)
    elif args.dataset == 'CIFAR10':
        args.channel = 3
        args.num_classes = 10
        args.im_size = (32, 32)
    elif args.dataset == 'CelebA':
        args.channel = 3
        args.num_classes = 2
        args.im_size = (64, 64)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

def randomize_labels(dataset):
    """
    Randomizes the labels of the given dataset.
    
    Args:
        dataset: Dataset whose labels need to be randomized.
    
    Returns:
        Dataset with randomized labels.
    """
    if hasattr(dataset, 'targets'):
        original_labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        original_labels = dataset.labels
    else:
        raise AttributeError("Dataset must have either 'targets' or 'labels' attribute.")

    shuffled_indices = torch.randperm(len(original_labels))
    randomized_labels = torch.tensor(original_labels)[shuffled_indices]

    if hasattr(dataset, 'targets'):
        dataset.targets = randomized_labels
    else:
        dataset.labels = randomized_labels

    return dataset

def main():
    # Parse arguments
    args = parse_args()

    # Set dataset-specific parameters
    set_dataset_params(args)

    # Set up the main logger
    logger = setup_main_logger(args.log_dir)
    logger.info("FedAvg Main Logger Initialized.")

    logger.info("Starting Federated Learning with %d clients", args.num_clients)

    # Load the training dataset
    train_dataset = load_data(args.dataset, data_path='/home/t914a431/data', train=True)

    # Check if partition_dir exists
    if not os.path.exists(args.partition_dir):
        logger.error(f"Partition directory {args.partition_dir} does not exist. Please run generate_partitions.py first.")
        raise FileNotFoundError(f"Partition directory {args.partition_dir} does not exist. Please run generate_partitions.py first.")

    # Load pre-partitioned data
    client_datasets_per_round = load_partitions(
        dataset=args.dataset,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        partition_dir=args.partition_dir,
        dataset_name=args.dataset,
        model_name=args.model,
        honesty_ratio=args.honesty_ratio
    )
    logger.info("Data partitions loaded successfully.")

    # Load the testing dataset
    test_dataset = load_data(args.dataset, data_path='/home/t914a431/data', train=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    logger.info("Test dataset loaded successfully.")

    # Initialize server
    server = Server(args)
    logger.info("Server initialized.")

    # Determine which clients are honest and which are dishonest
    random.seed(42)
    torch.manual_seed(42)
    num_honest_clients = int(args.honesty_ratio * args.num_clients)
    honest_clients = random.sample(range(args.num_clients), num_honest_clients)
    logger.info("Honest Clients: %s", honest_clients)
    clients = []

    # Initialize clients for the first round
    current_round = 0  # Zero-based indexing
    for i in range(args.num_clients):
        train_data = client_datasets_per_round[current_round][i]
        if i not in honest_clients:
            logger.info("Client %d is dishonest and will have randomized labels.", i)
            train_data = randomize_labels(train_data)  # Apply label randomization
        else:
            logger.info("Client %d is honest.", i)
        clients.append(Client(client_id=i, train_data=train_data, args=args))

    # Initialize list to store test accuracies
    test_accuracies = []

    # Federated learning rounds
    for round_num in range(1, args.num_rounds + 1):
        logger.info("\n--- Round %d ---", round_num)

        # Load partitioned data for each client for the current round
        current_round = round_num - 1  # Zero-based indexing
        client_datasets = client_datasets_per_round[current_round]

        # Update each client's data partition
        for client_id, client in enumerate(clients):
            client.train_data = client_datasets[client_id]
            if client_id not in honest_clients:
                logger.info("Client %d is dishonest and will have randomized labels for round %d.", client_id, round_num)
                client.train_data = randomize_labels(client.train_data)
            else:
                logger.info("Client %d is honest for round %d.", client_id, round_num)

        # Distribute the latest global model to clients
        global_model = server.get_global_model_state()
        
        # Move the global model state dictionary to CPU for serialization
        global_model_cpu = {key: value.cpu() for key, value in global_model.items()}
        
        # Clients perform local training in parallel using multiprocessing.Pool
        with multiprocessing.Pool(processes=args.num_clients) as pool:
            client_models = pool.starmap(
                train_client,
                [
                    (
                        client.client_id,
                        client.train_data,
                        global_model_cpu,
                        args,
                        client_id in honest_clients
                    )
                    for client_id, client in enumerate(clients)
                ]
            )

        logger.info("Clients have completed local training.")

        # Compute client sizes
        client_sizes = [len(client.train_data) for client in clients]
        logger.info("Client Sizes: %s", client_sizes)

        # Server aggregates client models
        server.aggregate(client_models, client_sizes)
        logger.info("Server has aggregated client models.")

        # Save the global model after aggregation
        server.save_model(round_num)
        logger.info("Global model saved after aggregation.")

        # Evaluate global model on test data
        accuracy = server.evaluate(test_loader, round_num)
        test_accuracies.append(accuracy)
        logger.info("Round %d: Global model accuracy: %.2f%%", round_num, accuracy)

    # Plot and save test accuracy graph
    plot_accuracies(
        test_accuracies=test_accuracies,
        model_name=args.model,
        dataset=args.dataset,
        alpha=args.alpha,
        num_partitions=args.num_clients,
        num_clients=args.num_clients,  # FedAvg specific
        save_dir=args.save_dir
    )

    logger.info("Federated Learning completed. Test accuracy graph saved.")

if __name__ == "__main__":
    main()
