# utils/generate_partitions.py

import argparse
from utils import load_data, partition_data_unique_rounds, save_partitions
import logging
import os
import matplotlib.pyplot as plt  # New import for plotting
import numpy as np  # New import for numerical operations
import matplotlib.cm as cm  # For color mapping

def setup_logger():
    """
    Sets up the logger for the partition generation script.
    """
    logger = logging.getLogger('GeneratePartitions')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        log_directory = "/home/t914a431/log"
        from utils.utils import ensure_directory_exists  # Import the utility function
        ensure_directory_exists(log_directory)
        file_handler = logging.FileHandler(os.path.join(log_directory, 'generate_partitions.log'))
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate Dataset Partitions for FedAF and FedAvg")
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'MNIST', 'CelebA'],
                        help='Dataset to use')
    parser.add_argument('--num_clients', type=int, required=True, help='Number of clients')
    parser.add_argument('--honesty_ratio', type=float, default=1.0, help='Honesty Ratio')  # Corrected default
    parser.add_argument('--model', type=str, default='ConvNet', help='Model')
    parser.add_argument('--num_rounds', type=int, required=True, help='Number of communication rounds')
    parser.add_argument('--alpha', type=float, default=0.1, help='Dirichlet distribution parameter for data heterogeneity')
    parser.add_argument('--data_path', type=str, default='/home/t914a431/data', help='Path to download/load the dataset')
    parser.add_argument('--save_dir', type=str, default='/home/t914a431/partitions_per_round', help='Directory to save data partitions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for multiprocessing')
    return parser.parse_args()

def plot_data_distribution_bubble(client_indices_per_round, save_path, num_clients, num_rounds):
    """
    Plots and saves the data distribution across clients for each round using a bubble chart.

    Args:
        client_indices_per_round (list): List containing client indices for each round.
        save_path (str): Path to save the plot.
        num_clients (int): Number of clients.
        num_rounds (int): Number of communication rounds.
    """
    # Initialize a numpy array to hold the counts
    data_counts = np.zeros((num_rounds, num_clients), dtype=int)
    
    for round_idx, client_indices in enumerate(client_indices_per_round):
        for client_idx, indices in enumerate(client_indices):
            data_counts[round_idx, client_idx] = len(indices)
    
    rounds = np.arange(1, num_rounds + 1)
    clients = np.arange(1, num_clients + 1)

    # Create a color map with sufficient distinct colors
    cmap = cm.get_cmap('hsv', num_clients)  # 'hsv' provides a wide range of colors
    colors = [cmap(i) for i in range(num_clients)]

    plt.figure(figsize=(16, 10))
    
    # Create a meshgrid for rounds and clients
    X, Y = np.meshgrid(rounds, clients)
    X = X.flatten()
    Y = Y.flatten()
    sizes = data_counts.flatten()

    # Normalize sizes for better visualization
    sizes_normalized = (sizes / sizes.max()) * 1000  # Adjust the multiplier as needed

    # Assign colors to each client
    client_colors = np.repeat(range(num_clients), num_rounds)
    color_values = [colors[client] for client in client_colors]

    scatter = plt.scatter(X, Y, s=sizes_normalized, c=client_colors, cmap='hsv', alpha=0.6, edgecolors='w', linewidth=0.5)

    plt.xlabel('Communication Rounds', fontsize=14)
    plt.ylabel('Clients', fontsize=14)
    plt.title('Data Distribution Across Clients per Round', fontsize=16)
    plt.xticks(ticks=np.arange(1, num_rounds + 1, max(1, num_rounds//10)))
    plt.yticks(ticks=np.arange(1, num_clients + 1, max(1, num_clients//10)))
    
    # Create a legend for clients
    handles = []
    for client_id in range(num_clients):
        handles.append(plt.Line2D([], [], marker='o', color=colors[client_id], linestyle='', markersize=10, label=f'Client {client_id+1}'))
    plt.legend(handles=handles, title="Clients", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    """
    Main function to generate and save dataset partitions.
    """
    # Initialize logging for multiprocessing
    logger = setup_logger()
    
    # Parse command-line arguments
    args = parse_args()
    
    # Log the received arguments
    logger.info(f"Received arguments: Dataset={args.dataset}, Num Clients={args.num_clients}, "
                f"Num Rounds={args.num_rounds}, Alpha={args.alpha}, Seed={args.seed}, "
                f"Num Workers={args.num_workers}")
    
    # Load the dataset
    logger.info(f"Loading dataset: {args.dataset} from {args.data_path}")
    dataset, labels = load_data(args.dataset, data_path=args.data_path, train=True)  # Receive labels
    logger.info(f"Dataset loaded with {len(dataset)} samples.")
    
    # Partition the dataset using multiprocessing
    logger.info(f"Starting dataset partitioning with {args.num_workers} workers.")
    client_indices_per_round = partition_data_unique_rounds(
        dataset=dataset,
        labels=labels,  # Pass labels to the partitioning function
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        alpha=args.alpha,
        seed=args.seed
    )
    logger.info("Dataset partitioning completed successfully.")
    
    # Save the partitions to the specified directory
    partitions_save_dir = os.path.join(
        args.save_dir,
        args.dataset,
        args.model,
        str(args.num_clients),
        str(args.honesty_ratio)
    )
    logger.info(f"Saving partitions to directory: {partitions_save_dir}")
    save_partitions(client_indices_per_round, partitions_save_dir, logger)
    logger.info(f"Data partitions generated and saved to {args.save_dir}")
    
    # Generate and save the data distribution graph
    graph_save_path = os.path.join(partitions_save_dir, 'data_distribution_bubble.png')
    logger.info(f"Generating data distribution bubble chart and saving to: {graph_save_path}")
    plot_data_distribution_bubble(
        client_indices_per_round=client_indices_per_round,
        save_path=graph_save_path,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds
    )
    logger.info("Data distribution bubble chart generated and saved successfully.")

if __name__ == "__main__":
    main()
