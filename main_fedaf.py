# main_fedaf.py

import os
import torch
import numpy as np
from client.client_fedaf import Client
from server.server_fedaf import server_update
from utils.utils_fedaf import get_dataset, get_network
from concurrent.futures import ThreadPoolExecutor, as_completed

class ARGS:
    def __init__(self):
        self.dataset = 'CIFAR10'  # or 'MNIST'
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
        self.Iteration = 1000
        self.lr_img = 1
        self.num_partitions = 5
        self.alpha = 0.1  # Dirichlet distribution parameter
        self.steps = 500
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
        elif self.dataset == 'CIFAR10':
            self.channel = 3
            self.num_classes = 10
            self.im_size = (32, 32)

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

    # Load and partition the dataset
    (
        channel,
        im_size,
        num_classes,
        class_names,
        mean,
        std,
        dst_train,
        dst_test,
        testloader,
    ) = get_dataset(
        dataset=args.dataset,
        data_path=args.data_path,
        num_partitions=args.num_partitions,
        alpha=args.alpha,
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

    # Prepare client IDs and data partitions
    client_ids = list(range(len(dst_train)))
    data_partitions = dst_train  # List of data partitions

    # Main communication rounds
    for r in range(1, rounds + 1):
        print(f"--- Round {r}/{rounds} ---")

        # Step 1: Clients calculate and save their Vkc logits (before data condensation)
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(client_calculate_and_save_V_logits, client_id, data_partitions[client_id], args, r)
                for client_id in client_ids
            ]
            for future in as_completed(futures):
                future.result()

        # Step 2: Server aggregates V logits and saves aggregated logits for clients
        aggregated_V_logits = aggregate_logits(client_ids, args.num_classes, 'V', args, r)
        save_aggregated_logits(aggregated_V_logits, args, r, 'V')

        # Step 3: Clients perform Data Condensation on synthetic data S
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(client_initialize_and_train_synthetic_data, client_id, data_partitions[client_id], args, r)
                for client_id in client_ids
            ]
            for future in as_completed(futures):
                future.result()

        # Step 4: Clients calculate and save their Rkc logits (after data condensation)
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(client_calculate_and_save_R_logits, client_id, args, r)
                for client_id in client_ids
            ]
            for future in as_completed(futures):
                future.result()

        # Step 5: Server aggregates R logits and saves aggregated logits
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

def client_calculate_and_save_V_logits(client_id, data_partition, args, r):
    """
    Function for clients to calculate and save V logits before data condensation.
    """
    client = Client(client_id, data_partition, args)
    client.calculate_and_save_V_logits(r)

def client_initialize_and_train_synthetic_data(client_id, data_partition, args, r):
    """
    Function for clients to initialize and train synthetic data.
    """
    client = Client(client_id, data_partition, args)
    client.initialize_synthetic_data(r)
    client.train_synthetic_data(r)
    client.save_synthetic_data(r)

def client_calculate_and_save_R_logits(client_id, args, r):
    """
    Function for clients to calculate and save R logits after data condensation.
    """
    client = Client(client_id, None, args)
    client.load_synthetic_data(r)
    client.calculate_and_save_R_logits(r)

def aggregate_logits(client_ids, num_classes, v_r, args, r):
    """
    Aggregates class-wise logits from all clients.
    """
    aggregated_logits = [torch.zeros(num_classes, device=args.device) for _ in range(num_classes)]
    count = [0 for _ in range(num_classes)]

    for client_id in client_ids:
        client_logit_path = os.path.join(args.logits_dir, f'Client_{client_id}', f'Round_{r}')
        for c in range(num_classes):
            client_Vkc_path = os.path.join(client_logit_path, f'{v_r}kc_{c}.pt')
            if os.path.exists(client_Vkc_path):
                client_logit = torch.load(client_Vkc_path, map_location=args.device)
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
            aggregated_logits[c] = torch.zeros(num_classes, device=args.device)  # Default if no clients have data for class c

    print(f"Server: Aggregated {v_r} logits computed.")
    return aggregated_logits

def save_aggregated_logits(aggregated_logits, args, r, v_r):
    """
    Saves the aggregated logits to a global directory accessible by all clients.
    """
    logits_dir = os.path.join(args.logits_dir, 'Global')
    os.makedirs(logits_dir, exist_ok=True)
    global_logits_path = os.path.join(logits_dir, f'Round{r}_Global_{v_r}c.pt')
    torch.save(aggregated_logits, global_logits_path)
    print(f"Server: Aggregated {v_r} logits saved to {global_logits_path}.")

if __name__ == '__main__':
    simulate(rounds=20)
