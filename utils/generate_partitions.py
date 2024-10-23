# utils/generate_partitions.py

import argparse
from utils import load_data, partition_data_per_round_parallel, save_partitions
import logging
import multiprocessing as mp
import os

def setup_logger():
    """
    Sets up the logger for the partition generation script.
    """
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        log_directory = "/home/t914a431/log"
        os.makedirs(log_directory, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_directory, 'generate_partitions.log'))
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s - %(message)s')
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
    parser.add_argument('--num_rounds', type=int, required=True, help='Number of communication rounds')
    parser.add_argument('--alpha', type=float, default=0.1, help='Dirichlet distribution parameter for data heterogeneity')
    parser.add_argument('--data_path', type=str, default='/home/t914a431/data', help='Path to download/load the dataset')
    parser.add_argument('--save_dir', type=str, default='/home/t914a431/partitions_per_round', help='Directory to save data partitions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for multiprocessing')
    return parser.parse_args()

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
    dataset = load_data(args.dataset, data_path=args.data_path)
    logger.info(f"Dataset loaded with {len(dataset)} samples.")
    
    # Partition the dataset using multiprocessing
    logger.info(f"Starting dataset partitioning with {args.num_workers} workers.")
    client_indices_per_round = partition_data_per_round_parallel(
        dataset=dataset,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        alpha=args.alpha,
        seed=args.seed,
        num_workers=args.num_workers
    )
    logger.info("Dataset partitioning completed successfully.")
    
    # Save the partitions to the specified directory
    logger.info(f"Saving partitions to directory: {args.save_dir}")
    save_partitions(client_indices_per_round, save_dir=args.save_dir)
    logger.info(f"Data partitions generated and saved to {args.save_dir}")

if __name__ == "__main__":
    main()
