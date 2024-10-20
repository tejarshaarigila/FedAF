# main_fedaf.py

import os
import torch
import multiprocessing
import numpy as np
from client.client_fedaf import Client
from server.server_fedaf import server_update
from utils.utils_fedaf import get_dataset, get_network
from multiprocessing import Pool 

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
    model = get_network(args.model, args.channel, args.num_classes, args.im_size).to(args.device)
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'fedaf_global_model_0.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Global model initialized and saved to {model_path}.")

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
    channel, im_size, num_classes, class_names, mean, std, dst_train, _, _ = get_dataset(
        dataset=args.dataset,
        data_path=args.data_path,
        num_partitions=args.num_partitions,
        alpha=args.alpha
    )
    args.channel = channel
    args.im_size = im_size
    args.num_classes = num_classes
    args.class_names = class_names
    args.mean = mean
    args.std = std

    # Initialize the global model and save it
    if not os.path.exists(os.path.join(args.model_dir, 'fedaf_global_model_0.pth')):
        print("[+] Initializing Global Model")
        initialize_global_model(args)

    # Initialize clients with their respective data partitions
    clients = []
    for client_id, data_partition in enumerate(dst_train):
        client = Client(client_id, data_partition, args)
        clients.append(client)

    # Main communication rounds
    for r in range(1, rounds + 1):
        print(f"---  Round: {r}/{rounds}  ---")

        # Step 1: Clients calculate and save their class-wise logits
        with Pool(processes=args.num_partitions) as pool:
            results = pool.map(calculate_and_save_logits_wrapper, [(client, r) for client in clients])

        # Prepare a mapping of client IDs to logit paths
        logit_paths = dict(results)

        # Step 2: Server aggregates logits and saves aggregated logits for clients
        aggregated_logits = aggregate_logits(logit_paths, args.num_classes, 'V')
        save_aggregated_logits(aggregated_logits, args, r, 'V')

        # Step 3: Clients perform Data Condensation on synthetic data S
        with Pool(processes=args.num_partitions) as pool:
            pool.map(data_condensation_wrapper, [(client, r) for client in clients])

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
            lambda_glob=args.loc_lgkm,
            hratio=args.honesty_ratio,
            temperature=args.temperature,
            num_epochs=args.global_steps,
            device=args.device
        )
        print(f"--- Round Ended: {r}/{rounds}  ---")

def calculate_and_save_logits_wrapper(args_tuple):
    client, r = args_tuple
    client.calculate_and_save_logits(r)
    return client.client_id, client.logit_path

def data_condensation_wrapper(args_tuple):
    client, r = args_tuple
    client.initialize_synthetic_data(r)
    client.train_synthetic_data(r)
    client.save_synthetic_data(r)

def aggregate_logits(logit_paths, num_classes, v_r):
    """
    Aggregates class-wise logits from all clients using their logit paths.

    Args:
        logit_paths (dict): Mapping from client_id to logit_path.
        num_classes (int): Number of classes.
        v_r (str): Indicator for the type of logits ('V' or 'R').

    Returns:
        list: Aggregated logits per class.
    """
    aggregated_logits = [torch.zeros(num_classes) for _ in range(num_classes)]
    count = [0 for _ in range(num_classes)]

    for client_id, client_logit_path in logit_paths.items():
        for c in range(num_classes):
            client_Vkc_path = os.path.join(client_logit_path, f'{v_r}kc_{c}.pt')
            if os.path.exists(client_Vkc_path):
                client_logit = torch.load(client_Vkc_path, map_location='cpu')
                if not torch.all(client_logit == 0):
                    aggregated_logits[c] += client_logit
                    count[c] += 1
            else:
                print(f"Server: Client {client_id} has no logits for class {c}. Skipping.")

    # Average the logits
    for c in range(num_classes):
        if count[c] > 0:
            aggregated_logits[c] /= count[c]
        else:
            aggregated_logits[c] = torch.zeros(num_classes)  # Default if no clients have data for class c

    print("Server: Aggregated logits computed.")
    return aggregated_logits

def save_aggregated_logits(aggregated_logits, args, r, v_r):
    """
    Saves the aggregated logits to a global directory accessible by all clients.
    """
    logits_dir = os.path.join(args.logits_dir, 'Global')
    os.makedirs(logits_dir, exist_ok=True)
    global_logits_path = os.path.join(logits_dir, f'Round{r}_Global_{v_r}c.pt')
    torch.save(aggregated_logits, global_logits_path)
    print(f"Server: Aggregated logits saved to {global_logits_path}.")

if __name__ == '__main__':

    multiprocessing.set_start_method('spawn', force=True)  # Ensure compatibility across platforms
    simulate(rounds=50)
