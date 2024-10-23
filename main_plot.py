# main_plot.py

import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from utils.utils import get_network
import numpy as np
from multiprocessing import Pool, set_start_method
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s',
    filename='/home/t914a431/log/plotting.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

class PlotArgs:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Federated Learning Plotting Script')
        
        # Required Arguments
        parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'MNIST'],
                            help='Dataset name (CIFAR10 or MNIST)')
        parser.add_argument('--model', type=str, default='ConvNet',
                            help='Model architecture (e.g., ConvNet)')
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                            help='Device to use (cuda or cpu)')
        parser.add_argument('--test_repeats', type=int, default=5,
                            help='Number of times to repeat testing for averaging')
        parser.add_argument('--num_users', type=int, default=10,
                            help='Number of clients')
        parser.add_argument('--honesty_ratio', type=float, default=1.0,
                            help='Honesty ratio parameter')
        parser.add_argument('--alpha_dirichlet', type=float, default=0.1,
                            help='Dirichlet distribution parameter alpha')
        
        # Optional Arguments
        parser.add_argument('--methods', type=str, nargs='+', default=['fedaf', 'fedavg'],
                            help='Methods to compare (e.g., fedaf fedavg)')
        parser.add_argument('--model_base_dir', type=str, default='/home/t914a431/models',
                            help='Base directory for models')
        parser.add_argument('--save_dir', type=str, default='/home/t914a431/results/',
                            help='Directory to save the plots')
        
        args = parser.parse_args()
        
        self.dataset = args.dataset
        self.model = args.model
        self.device = args.device
        self.test_repeats = args.test_repeats
        self.num_users = args.num_users
        self.honesty_ratio = args.honesty_ratio
        self.alpha_dirichlet = args.alpha_dirichlet
        self.methods = args.methods
        self.save_dir = args.save_dir
        
        # Set default model_base_dir if not provided
        self.model_base_dir = os.path.join(args.model_base_dir, self.dataset, self.model, str(self.num_users), str(self.honesty_ratio))
        
        # Set dataset-specific parameters
        if self.dataset == 'MNIST':
            self.channel = 1
            self.num_classes = 10
            self.im_size = (28, 28)
            self.mean = [0.1307]
            self.std = [0.3081]
        elif self.dataset == 'CIFAR10':
            self.channel = 3
            self.num_classes = 10
            self.im_size = (32, 32)
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.2023, 0.1994, 0.2010]

def load_test_dataset(args):
    """Loads the test dataset based on the given dataset name."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ])

    if args.dataset == 'MNIST':
        test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    elif args.dataset == 'CIFAR10':
        test_dataset = datasets.CIFAR10('data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    return test_loader

def evaluate_model(model, test_loader, device):
    """Evaluates the model on the test data and returns accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # No gradients needed
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def evaluate_model_wrapper(args_tuple):
    method, model_file, round_num, args = args_tuple
    model_dir = os.path.join(args.model_base_dir)
    model_path = os.path.join(model_dir, model_file)
    
    if not os.path.exists(model_path):
        logger.warning(f"Model file does not exist: {model_path}")
        return method, round_num, None  # Indicate missing model
    
    # Instantiate the model
    model = get_network(args.model, args.channel, args.num_classes, args.im_size, device=args.device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=args.device))
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {e}")
        return method, round_num, None
    
    # Create the test loader
    test_loader = load_test_dataset(args)
    
    # Evaluate the model multiple times
    round_accuracies = [evaluate_model(model, test_loader, args.device) for _ in range(args.test_repeats)]
    avg_accuracy = np.mean(round_accuracies)
    logger.info(f"Method: {method}, Round {round_num}: Avg Test Accuracy = {avg_accuracy:.2f}%")
    return method, round_num, avg_accuracy

def test_saved_models(args):
    # Collect all (method, model_file, round_num) tuples to evaluate
    eval_tasks = []

    for method in args.methods:
        model_dir = os.path.join(args.model_base_dir)
        if os.path.exists(model_dir):
            model_files = sorted(
                [f for f in os.listdir(model_dir) if f.startswith(f"{method}_global_model") and f.endswith('.pth')],
                key=lambda x: int(x.split('_')[-1].split('.')[0])  # Extract round_num from filename
            )
        else:
            logger.warning(f"Directory does not exist: {model_dir}")
            continue

        for model_file in model_files:
            # Extract round number from filename, assuming format "{method}_global_model_{round_num}.pth"
            try:
                round_num_str = model_file.split('_')[-1].split('.')[0]
                round_num = int(round_num_str)
            except (IndexError, ValueError):
                logger.error(f"Invalid model filename format: {model_file}")
                continue
            
            eval_tasks.append((method, model_file, round_num, args))

    method_accuracies = {method: {} for method in args.methods}

    try:
        set_start_method('spawn', force=True)  # Ensure compatibility across platforms
    except RuntimeError:
        pass  # If the start method has already been set

    with Pool(processes=min(args.num_users, os.cpu_count())) as pool:
        results = pool.map(evaluate_model_wrapper, eval_tasks)

    # Collect the results
    for method, round_num, avg_accuracy in results:
        if avg_accuracy is not None:
            method_accuracies[method][round_num] = avg_accuracy
        else:
            logger.warning(f"Skipping plot for Method: {method}, Round: {round_num} due to missing accuracy.")

    # Plotting the test accuracies for each method
    plt.figure(figsize=(12, 8))

    for method, accuracies in method_accuracies.items():
        if not accuracies:
            logger.warning(f"No accuracies recorded for method: {method}. Skipping plot.")
            continue
        # Sort rounds and corresponding accuracies
        rounds = sorted(accuracies.keys())
        test_accuracies = [accuracies[r] for r in rounds]

        # Plot accuracies for the current method
        plt.plot(rounds, test_accuracies, marker='o', linestyle='-', label=f"{method.upper()}")

    plt.title(f"Dataset: {args.dataset}, Users: {args.num_users}, Alpha: {args.alpha_dirichlet}")
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    
    if rounds:
        plt.xticks(rounds)
    
    plt.legend()

    # Ensure the save directory exists
    ensure_save_directory = os.path.join(args.save_dir)
    os.makedirs(ensure_save_directory, exist_ok=True)

    # Save and show plot
    plot_save_path = os.path.join(
        args.save_dir,
        f"{args.dataset}_{args.model}_C{args.num_users}_alpha{args.alpha_dirichlet}.png"
    )
    plt.savefig(plot_save_path)
    plt.close()  # Close the plot to free memory
    logger.info(f"Plot saved to {plot_save_path}")

def main():
    args = PlotArgs()
    test_saved_models(args)

if __name__ == "__main__":
    main()
