# main_fedavg.py

import torch
import torch.multiprocessing as multiprocessing
from server.server_fedavg import Server
from client.client_fedavg import Client
from utils.utils_fedavg import load_data
from utils.plot_utils import plot_accuracies
import logging
import random
import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='fedavg.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

class ARGS:
    def __init__(self):
        self.dataset = 'CIFAR10'  # 'MNIST' - 'CIFAR10' - 'CelebA'
        self.model = 'ConvNet'     # 'ConvNet' - 'ResNet'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_clients = 10
        self.alpha = 0.1  # Dirichlet distribution parameter
        self.local_epochs = 1
        self.lr = 0.01
        self.batch_size = 64
        self.num_rounds = 50
        self.honesty_ratio = 1  # Ratio of Honest Clients

        if self.dataset == 'MNIST':
            self.channel = 1
            self.num_classes = 10
            self.im_size = (28, 28)
        elif self.dataset == 'CIFAR10':
            self.channel = 3
            self.num_classes = 10
            self.im_size = (32, 32)
        elif self.dataset == 'CelebA':
            self.channel = 3
            self.num_classes = 2
            self.im_size = (64, 64)

def randomize_labels(dataset):
    """Randomly switch labels of the dataset."""
    randomized_dataset = copy.deepcopy(dataset)
    labels = randomized_dataset.targets if hasattr(randomized_dataset, 'targets') else randomized_dataset.labels
    shuffled_labels = torch.randperm(len(labels))
    labels = labels[shuffled_labels]
    if hasattr(randomized_dataset, 'targets'):
        randomized_dataset.targets = labels
    else:
        randomized_dataset.labels = labels
    return randomized_dataset

def client_train_worker(args_tuple):
    client_id, train_data, args_dict, global_model_state = args_tuple
    try:
        # Reconstruct ARGS instance from args_dict
        args = ARGS()
        args.__dict__.update(args_dict)
        # Initialize the client inside the worker process
        client = Client(client_id, train_data, args)
        client.set_model(global_model_state)
        client_model = client.train()
        client_size = len(client.train_loader.dataset)
        logger.info(f"Client {client_id} finished training.")
        return client_model, client_size
    except Exception as e:
        logger.exception(f"Exception in client {client_id}: {e}")
        return None, 0

def main():
    args = ARGS()
    logger.info("Starting Federated Learning with %d clients", args.num_clients)

    # Load data and distribute to clients
    client_datasets, test_loader = load_data(
        dataset=args.dataset,
        alpha=args.alpha,
        num_clients=args.num_clients
    )
    logger.info("Data loaded and distributed to clients.")

    # Initialize server
    server = Server(args)

    # Determine which clients are honest and which are dishonest
    num_honest_clients = int(args.honesty_ratio * args.num_clients)
    honest_clients = random.sample(range(args.num_clients), num_honest_clients)

    # Prepare arguments for each client
    args_dict = vars(args)  # Convert ARGS instance to a dictionary

    # Lists to collect test accuracies
    test_accuracies = []

    # Federated learning rounds
    for round_num in range(1, args.num_rounds + 1):
        logger.info("\n--- Round %d ---", round_num)

        # Get the global model state
        global_model_state = {k: v.cpu() for k, v in server.get_global_model_state().items()}

        # Prepare client arguments for multiprocessing
        client_args = []
        for client_id, train_data in enumerate(client_datasets):
            if client_id not in honest_clients:
                logger.info("Client %d is dishonest and will have randomized labels.", client_id)
                train_data = randomize_labels(train_data)  # Apply label switching
            client_args.append((client_id, train_data, args_dict, global_model_state))

        # Clients perform local training in parallel
        with multiprocessing.Pool(processes=args.num_clients) as pool:
            results = pool.map(client_train_worker, client_args)

        client_models, client_sizes = zip(*results)

        # Server aggregates client models
        server.aggregate(client_models, client_sizes)

        # Evaluate global model on test data and save the model with the current round number
        accuracy = server.evaluate(test_loader, round_num)
        test_accuracies.append(accuracy)

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
    multiprocessing.set_start_method('spawn', force=True)
    main()
