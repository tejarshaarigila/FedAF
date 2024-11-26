# main_fedaf.py

import os
import copy
import time
import numpy as np
import torch
from torch.utils.data import Subset
from concurrent.futures import ProcessPoolExecutor, as_completed
from client.client_fedaf import Client
from server.server_fedaf import server_update
from utils.utils_fedaf import get_network, get_base_dataset, save_aggregated_logits, ensure_directory_exists

class ARGS:
    def __init__(self):
        self.dataset = 'CIFAR10'  # or 'CIFAR10'
        self.model = 'ConvNet'
        self.model_name = self.model
        self.method = 'DM'
        self.init = 'real'
        self.data_path = '/home/t914a431/data'
        self.logits_dir = '/home/t914a431/logits'
        self.save_image_dir = '/home/t914a431/images'
        self.save_path = '/home/t914a431/result'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ipc = 50  # Instances Per Class
        self.eval_mode = 'SS'
        self.Iteration = 1000  # Local Steps
        self.lr_img = 1
        self.num_partitions = 15
        self.alpha = 0.1  # Dirichlet distribution parameter
        self.steps = 500  # Global Steps
        self.loc_cdc = 0.8
        self.loc_lgkm = 0.8
        self.temperature = 2.0
        self.gamma = 0.9
        self.honesty_ratio = 1
        self.model_dir = f'/home/t914a431/models/{self.dataset}/{self.model}/{self.num_partitions}/{self.honesty_ratio}'
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
            
    @classmethod
    def from_dict(cls, args_dict):
        args = cls()
        for key, value in args_dict.items():
            setattr(args, key, value)
        return args

def initialize_global_model(args):
    """
    Initializes a random global model and saves it so that clients can access it.
    """
    model = get_network(args.model, args.channel, args.num_classes, args.im_size)
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'fedaf_global_model_0.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Global model initialized and saved to {model_path}.")

def simulate(rounds):
    args = ARGS()
    args.eval_it_pool = (
        np.arange(0, args.Iteration + 1, args.steps).tolist()
        if args.eval_mode in ['S', 'SS']
        else [args.Iteration]
    )  # The list of iterations when we evaluate models and record results.

    # Create necessary directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.logits_dir, exist_ok=True)
    os.makedirs(args.save_image_dir, exist_ok=True)

    # Load and partition the dataset once
    base_dataset = get_base_dataset(args.dataset, args.data_path, train=True)
    labels = np.array(base_dataset.targets)
    indices = [[] for _ in range(args.num_partitions)]

    # Partitioning logic
    num_classes = args.num_classes
    num_partitions = args.num_partitions
    alpha = args.alpha

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet(np.repeat(alpha, num_partitions))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        class_splits = np.split(class_indices, proportions)
        for idx in range(num_partitions):
            if idx < len(class_splits):
                indices[idx].extend(class_splits[idx])

    data_partition_indices = indices  # Use indices directly

    # Initialize the global model and save it
    if not os.path.exists(os.path.join(args.model_dir, 'fedaf_global_model_0.pth')):
        print("[+] Initializing Global Model")
        initialize_global_model(args)

    # Prepare client IDs
    client_ids = list(range(args.num_partitions))

    # Prepare picklable arguments
    args_dict = copy.deepcopy(vars(args))
    args_dict['device'] = str(args.device)  # Convert torch.device to string

    # Main communication rounds
    for r in range(1, rounds + 1):
        print(f"--- Round {r}/{rounds} ---")

        # Phase 1: Clients compute Vkc
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    client_compute_Vkc,
                    client_id,
                    data_partition_indices[client_id],
                    args_dict,
                    r,
                    base_dataset
                )
                for client_id in client_ids
            ]
            # Wait for all clients to finish computing Vkc
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in client_compute_Vkc: {e}")

        # Server aggregates V logits and saves aggregated logits for clients
        aggregated_V_logits = aggregate_logits(client_ids, args.num_classes, 'V', args, r)
        save_aggregated_logits(aggregated_V_logits, args, r, 'V')

        # Phase 2: Clients perform data condensation and compute Rkc
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    client_data_condensation_and_Rkc,
                    client_id,
                    data_partition_indices[client_id],
                    args_dict,
                    r,
                    base_dataset
                )
                for client_id in client_ids
            ]
            # Wait for all clients to finish data condensation and Rkc computation
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in client_data_condensation_and_Rkc: {e}")

        # Server aggregates R logits and saves aggregated logits
        aggregated_R_logits = aggregate_logits(client_ids, args.num_classes, 'R', args, r)
        save_aggregated_logits(aggregated_R_logits, args, r, 'R')

        # Step 6: Server updates the global model using aggregated R logits and synthetic data
        server_update(
            model_name=args.model,
            data=args.dataset,
            num_partitions=args.num_partitions,
            round_num=r,
            lambda_glob=args.loc_lgkm,
            ipc=args.ipc,
            method=args.method,
            hratio=args.honesty_ratio,
            temperature=args.temperature,
            num_epochs=args.steps,
            device=args.device,
        )

def client_compute_Vkc(client_id, data_partition_indices, args_dict, r, base_dataset):
    # Reconstruct args
    args = ARGS.from_dict(args_dict)
    args.device = torch.device(args.device)

    # Reconstruct data_partition
    data_partition = Subset(base_dataset, data_partition_indices)

    # Create client instance
    client = Client(client_id, data_partition, args)

    # Load and resample the model once
    client.model = client.load_global_model()
    client.resample_model(client.model)

    # Run Vkc computation
    client.run_Vkc(r)

def client_data_condensation_and_Rkc(client_id, data_partition_indices, args_dict, r, base_dataset):
    # Reconstruct args
    args = ARGS.from_dict(args_dict)
    args.device = torch.device(args.device)

    # Reconstruct data_partition
    data_partition = Subset(base_dataset, data_partition_indices)

    # Create client instance
    client = Client(client_id, data_partition, args)

    # Load the model (if necessary)
    client.model = client.load_global_model()
    client.resample_model(client.model)

    # Wait for the server to aggregate Vc
    print(f"Client {client_id} is waiting for global Vc aggregation.")
    global_Vc_path = os.path.join(args.logits_dir, 'Global', f'Round{r}_Global_Vc.pt')
    while not os.path.exists(global_Vc_path):
        time.sleep(1)  # Wait for 1 second before checking again
    print(f"Client {client_id} received signal to proceed.")

    # Proceed to data condensation and Rkc computation
    client.run_data_condensation(r)
    client.run_Rkc(r)

def aggregate_logits(client_ids, num_classes, v_r, args, r):
    """
    Aggregates class-wise logits from all clients into a tensor of shape [num_classes,].

    Args:
        client_ids (list): List of client IDs.
        num_classes (int): Number of classes.
        v_r (str): Type of logits ('V' or 'R').
        args (ARGS): Argument parser containing configurations.
        r (int): Current round number.

    Returns:
        torch.Tensor: Aggregated logits Rc of shape [num_classes,].
    """
    aggregated_logits = torch.zeros(num_classes, device=args.device)
    count = torch.zeros(num_classes, device=args.device)

    for client_id in client_ids:
        client_logit_path = os.path.join(args.logits_dir, f'Client_{client_id}', f'Round_{r}')
        for c in range(num_classes):
            client_Vkc_path = os.path.join(client_logit_path, f'{v_r}kc_{c}.pt')
            if os.path.exists(client_Vkc_path):
                try:
                    client_logit = torch.load(client_Vkc_path, map_location=args.device)
                    if isinstance(client_logit, torch.Tensor):
                        if client_logit.numel() == num_classes:
                            # Accumulate the logit value at index c
                            aggregated_logits[c] += client_logit[c]
                            count[c] += 1
                        else:
                            print(f"Server: Client {client_id} logits for class {c} have incorrect number of elements. Expected {num_classes}, got {client_logit.numel()}. Skipping.")
                    else:
                        print(f"Server: Client {client_id} logits for class {c} are not tensors. Skipping.")
                except Exception as e:
                    print(f"Server: Failed to load logits for Client {client_id}, Class {c} - {e}")
            else:
                print(f"Server: Client {client_id} has no logits for class {c}. Skipping.")

    # Avoid division by zero
    count = torch.where(count == 0, torch.ones_like(count), count)

    # Average the logits per class
    Rc = aggregated_logits / count  # Shape: [num_classes,]

    print(f"Server: Aggregated {v_r} logits computed with Rc shape {Rc.shape}.")

    return Rc

def save_aggregated_logits(aggregated_logits, args, r, v_r):
    """
    Saves the aggregated logits to a global directory accessible by all clients.

    Args:
        aggregated_logits (torch.Tensor): Aggregated logits Rc of shape [num_classes,].
        args (ARGS): Argument parser containing configurations.
        r (int): Current round number.
        v_r (str): Type of logits ('V' or 'R').
    """
    logits_dir = os.path.join(args.logits_dir, 'Global')
    os.makedirs(logits_dir, exist_ok=True)
    global_logits_path = os.path.join(logits_dir, f'Round{r}_Global_{v_r}c.pt')
    torch.save(aggregated_logits, global_logits_path)  # Saving single tensor
    print(f"Server: Aggregated {v_r} logits saved to {global_logits_path}.")

if __name__ == '__main__':
    simulate(rounds=20)
