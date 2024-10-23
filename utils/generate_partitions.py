# utils/generate_partitions.py

import argparse
from utils import load_data, partition_data_per_round, save_partitions

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Dataset Partitions for FedAF and FedAvg")
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'MNIST', 'CelebA'],
                        help='Dataset to use')
    parser.add_argument('--num_clients', type=int, required=True, help='Number of clients')
    parser.add_argument('--num_rounds', type=int, required=True, help='Number of communication rounds')
    parser.add_argument('--alpha', type=float, default=0.1, help='Dirichlet distribution parameter for data heterogeneity')
    parser.add_argument('--data_path', type=str, default='/home/t914a431/data', help='Path to download/load the dataset')
    parser.add_argument('--save_dir', type=str, default='/home/t914a431/partitions_per_round', help='Directory to save data partitions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def main():
    args = parse_args()
    dataset = load_data(args.dataset, data_path=args.data_path)
    client_indices_per_round = partition_data_per_round(
        dataset=dataset,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        alpha=args.alpha,
        seed=args.seed
    )
    save_partitions(client_indices_per_round, save_dir=args.save_dir)
    print(f"Data partitions generated and saved to {args.save_dir}")

if __name__ == "__main__":
    main()
