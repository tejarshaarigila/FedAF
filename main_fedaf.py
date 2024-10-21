# main_fedaf.py

import os
import torch
import numpy as np
import logging
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

class ARGS:
    def __init__(self):
        self.dataset = 'CIFAR10'  # or 'MNIST'
        self.model = 'ConvNet'
        self.model_name = self.model
        self.method = 'DM'
        self.init = 'real'
        self.data_path = 'data'
        self.logits_dir = 'logits'
        self.save_image_dir = 'images'
        self.save_path = 'result'
        self.device = 'cpu'
        self.rounds = 50
        self.ipc = 50  # Instances Per Class
        self.eval_mode = 'SS'
        self.Iteration = 500
        self.lr_img = 1
        self.num_partitions = 10
        self.alpha = 0.1  # Dirichlet distribution parameter
        self.steps = 1000
        self.global_steps = 500
        self.loc_cdc = 0.8
        self.loc_lgkm = 0.8
        self.temperature = 2.0
        self.gamma = 0.9
        self.honesty_ratio = 1
        self.num_workers = 0  # Set to 0 for multiprocessing compatibility
        self.model_dir = f'./models/{self.dataset}/{self.model}/{self.num_partitions}/{self.honesty_ratio}'
        if self.dataset == 'MNIST':
            self.channel = 1
            self.num_classes = 10
            self.im_size = (28,28)
        elif self.dataset == 'CIFAR10':
            self.channel = 3
            self.num_classes = 10
            self.im_size = (32,32)
        self.eval_it_pool = np.arange(0, self.Iteration + 1, self.steps).tolist() if self.eval_mode in ['S', 'SS'] else [self.Iteration]

def initialize_global_model(args):
    """
    Initializes a random global model and saves it so that clients can access it.
    """
    model = get_network(args.model, args.channel, args.num_classes, args.im_size, device=args.device)
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'fedaf_global_model_0.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Global model initialized and saved to {model_path}.")

def simulate():
    args = ARGS()
    rounds = args.rounds

    # Create necessary directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.logits_dir, exist_ok=True)
    os.makedirs(args.save_image_dir, exist_ok=True)

    # Load and partition the dataset
    client_datasets, test_loader = load_data(
        dataset=args.dataset,
        alpha=args.alpha,
        num_clients=args.num_partitions,
        seed=42  # For reproducibility
    )
    args.channel = client_datasets[0][0][0].shape[0]
    args.im_size = client_datasets[0][0][0].shape[1:]
    args.num_classes = len(np.unique([label for _, label in client_datasets[0]]))

    # Initialize the global model and save it
    if not os.path.exists(os.path.join(args.model_dir, 'fedaf_global_model_0.pth')):
        logger.info("[+] Initializing Global Model")
        initialize_global_model(args)

    args_dict = vars(args)  # Convert ARGS instance to a dictionary

    # Main communication rounds
    for r in range(1, rounds + 1):
        logger.info(f"---  Round: {r}/{rounds}  ---")

        # Step 1: Clients calculate and save their class-wise logits
        client_args = [
            (client_id, train_data, args_dict, r)
            for client_id, train_data in enumerate(client_datasets)
        ]
        with multiprocessing.Pool(processes=args.num_partitions) as pool:
            logit_paths = pool.map(calculate_and_save_logits_worker, client_args)

        # Step 2: Server aggregates logits and saves aggregated logits for clients
        aggregated_logits = aggregate_logits(logit_paths, args.num_classes, 'V')
        save_aggregated_logits(aggregated_logits, args, r, 'V')

        # Step 3: Clients perform Data Condensation on synthetic data S
        client_args = [
            (client_id, train_data, args_dict, r)
            for client_id, train_data in enumerate(client_datasets)
        ]
        with multiprocessing.Pool(processes=args.num_partitions) as pool:
            pool.map(data_condensation_worker, client_args)

        # Step 4: Server updates the global model using aggregated soft labels R & synthetic data S
        aggregated_labels = aggregate_logits(logit_paths, args.num_classes, 'R')
        save_aggregated_logits(aggregated_labels, args, r, 'R')
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
    client_id, train_data, args_dict, r = args_tuple
    try:
        # Reconstruct ARGS instance
        args = ARGS()
        args.__dict__.update(args_dict)
        # Initialize Client
        client = Client(client_id, train_data, args)
        client.calculate_and_save_logits(r)
        # Log when client completes calculating logits
        logger.info(f"Client {client_id} has completed calculating and saving logits for round {r}.")
        return client.logit_path
    except Exception as e:
        logger.exception(f"Exception in client {client_id} during logits calculation: {e}")
        return None

def data_condensation_worker(args_tuple):
    client_id, train_data, args_dict, r = args_tuple
    try:
        # Reconstruct ARGS instance
        args = ARGS()
        args.__dict__.update(args_dict)
        # Initialize Client
        client = Client(client_id, train_data, args)
        client.initialize_synthetic_data(r)
        client.train_synthetic_data(r)
        client.save_synthetic_data(r)
        # Log when client completes data condensation
        logger.info(f"Client {client_id} has completed data condensation for round {r}.")
    except Exception as e:
        logger.exception(f"Exception in client {client_id} during data condensation: {e}")

def aggregate_logits(logit_paths, num_classes, v_r):
    """
    Aggregates class-wise logits from all clients using their logit paths.

    Args:
        logit_paths (list): List of logit paths from clients.
        num_classes (int): Number of classes.
        v_r (str): Indicator for the type of logits ('V' or 'R').

    Returns:
        list: Aggregated logits per class.
    """
    aggregated_logits = [torch.zeros(num_classes) for _ in range(num_classes)]
    count = [0 for _ in range(num_classes)]

    for client_logit_path in logit_paths:
        if client_logit_path is None:
            continue
        for c in range(num_classes):
            client_Vkc_path = os.path.join(client_logit_path, f'{v_r}kc_{c}.pt')
            if os.path.exists(client_Vkc_path):
                client_logit = torch.load(client_Vkc_path, map_location='cpu', weights_only = True)
                if not torch.all(client_logit == 0):
                    aggregated_logits[c] += client_logit
                    count[c] += 1
            else:
                logger.warning(f"Server: Missing logits for class {c} in {client_logit_path}. Skipping.")

    # Average the logits
    for c in range(num_classes):
        if count[c] > 0:
            aggregated_logits[c] /= count[c]
        else:
            aggregated_logits[c] = torch.zeros(num_classes)  # Default if no clients have data for class c

    logger.info("Server: Aggregated logits computed.")
    return aggregated_logits

def save_aggregated_logits(aggregated_logits, args, r, v_r):
    """
    Saves the aggregated logits to a global directory accessible by all clients.
    """
    logits_dir = os.path.join(args.logits_dir, 'Global')
    os.makedirs(logits_dir, exist_ok=True)
    global_logits_path = os.path.join(logits_dir, f'Round{r}_Global_{v_r}c.pt')
    torch.save(aggregated_logits, global_logits_path)
    logger.info(f"Server: Aggregated logits saved to {global_logits_path}.")

if __name__ == '__main__':
    simulate()
