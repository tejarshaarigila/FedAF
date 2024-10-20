import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.utils_fedaf import get_network
import numpy as np
from multiprocessing import Pool, set_start_method
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='plotting.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

class PlotArgs:
    def __init__(self):
        self.dataset = 'CIFAR10'  # 'CIFAR10','MNIST'
        self.model = 'ConvNet'  # 'ConvNet'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.test_repeats = 5  # Number of times to repeat testing for averaging
        self.num_user = 10  # Number of clients
        self.honesty_ratio = 1  # Honesty ratio parameter
        self.methods = ['fedaf', 'fedavg']  # Methods to compare
        self.model_base_dir = f'./models/{self.dataset}/{self.model}/{self.num_user}/{self.honesty_ratio}/'

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
    with torch.no_grad():
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
    # Instantiate the model
    model = get_network(args.model, args.channel, args.num_classes, args.im_size, device=args.device)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
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
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
        else:
            logger.warning(f"Directory does not exist: {model_dir}")
            continue

        for model_file in model_files:
            round_num = int(model_file.split('_')[-1].split('.')[0])
            if method == 'fedaf':
                round_num += 1
            eval_tasks.append((method, model_file, round_num, args))

    method_accuracies = {method: {} for method in args.methods}

    try:
        set_start_method('spawn', force=True)  # Ensure compatibility across platforms
    except RuntimeError:
        pass  # If the start method has already been set

    with Pool(processes=args.num_user) as pool:
        results = pool.map(evaluate_model_wrapper, eval_tasks)

    # Collect the results
    for method, round_num, avg_accuracy in results:
        method_accuracies[method][round_num] = avg_accuracy

    # Plotting the test accuracies for each method
    plt.figure(figsize=(12, 8))

    for method, accuracies in method_accuracies.items():
        # Sort rounds and corresponding accuracies
        rounds = sorted(accuracies.keys())
        test_accuracies = [accuracies[r] for r in rounds]

        # Plot accuracies for the current method
        plt.plot(rounds, test_accuracies, marker='o', linestyle='-', label=f"{method.upper()}")

    plt.title(f"Test Accuracy over Communication Rounds\nDataset: {args.dataset}, Honesty Ratio: {args.honesty_ratio}")
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.xticks(rounds)
    plt.legend()

    # Save and show plot
    plot_save_path = f"plots/test_accuracy_comparison_{args.dataset}_honesty_{args.honesty_ratio}.png"
    os.makedirs('plots', exist_ok=True)
    plt.savefig(plot_save_path)
    plt.show()
    logger.info(f"Plot saved to {plot_save_path}")

if __name__ == "__main__":
    args = PlotArgs()
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    test_saved_models(args)
