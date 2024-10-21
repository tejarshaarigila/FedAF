# main_fedavg.py

import os
import argparse
import torch
from server.server_fedavg import Server
from client.client_fedavg import Client
from utils.utils_fedavg import load_data
from utils.plot_utils import plot_accuracies
import logging
import random
import concurrent.futures

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
    return parser.parse_args()

def train_client(client):
    """Helper function to train a client."""
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
    # Ensure the log directory exists
    log_dir = "/home/t914a431/log/"
    os.makedirs(log_dir, exist_ok=True)

    # Set up the main logger
    logger = setup_main_logger(log_dir)
    logger.info("FedAvg Main Logger Initialized.")

    args = parse_args()
    set_dataset_params(args)
    logger.info("Starting Federated Learning with %d clients", args.num_clients)

    # Load data and distribute to clients
    client_datasets, test_loader = load_data(dataset=args.dataset, alpha=args.alpha, num_clients=args.num_clients)
    logger.info("Data loaded and distributed to clients.")

    # Initialize server and clients
    server = Server(args)
    logger.info("Server initialized.")

    # Determine which clients are honest and which are dishonest
    random.seed(42)
    torch.manual_seed(42)
    num_honest_clients = int(args.honesty_ratio * args.num_clients)
    honest_clients = random.sample(range(args.num_clients), num_honest_clients)
    logger.info("Honest Clients: %s", honest_clients)
    clients = []

    for i in range(args.num_clients):
        train_data = client_datasets[i]
        if i not in honest_clients:
            logger.info("Client %d is dishonest and will have randomized labels.", i)
            train_data = randomize_labels(train_data)  # Apply label switching
        else:
            logger.info("Client %d is honest.", i)
        clients.append(Client(client_id=i, train_data=train_data, args=args))

    # Lists to collect test accuracies
    test_accuracies = []

    # Federated learning rounds
    for round_num in range(1, args.num_rounds + 1):
        logger.info("\n--- Round %d ---", round_num)

        # Distribute global model to clients
        global_model = server.get_global_model_state()
        for client in clients:
            client.set_model(global_model)
        logger.info("Global model distributed to clients.")

        # Clients perform local training in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_clients) as executor:
            # Execute training
            client_models = list(executor.map(train_client, clients))
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
        num_clients=args.num_clients
    )

    logger.info("Federated Learning completed. Test accuracy graph saved.")

if __name__ == "__main__":
    main()
