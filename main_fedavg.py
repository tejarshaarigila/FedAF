# main_fedavg.py
#
# FedAvg baseline driver. Settings come from config.yaml via
# config.load_config('fedavg'), so dataset / model / client-count / paths are
# shared with FedAF and the plotter (configure them in exactly one place).
#
# Restartability (cfg.resume): the run resumes from the last saved round
# checkpoint (models/.../fedavg_global_model_{r}.pth); completed rounds are
# skipped and the global model is restored before continuing.

import torch
import logging
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from config import load_config, Config, latest_completed_round
from server.server_fedavg import Server
from client.client_fedavg import Client
from utils.utils_fedavg import load_data, randomize_labels, load_client_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_client(client_state):
    client_id, cfg_dict, global_model_state, is_dishonest = client_state
    cfg = Config.from_dict(cfg_dict)
    # Load client's data (partition is seeded -> identical to the server's split)
    train_data = load_client_data(client_id, cfg)
    # Apply label randomization if the client is dishonest
    if is_dishonest:
        train_data = randomize_labels(train_data)
    # Reconstruct Client
    client = Client(client_id=client_id, train_data=train_data, args=cfg)
    client.set_model(global_model_state)
    return client.train()


def main(cfg=None):
    cfg = cfg or load_config('fedavg')
    # Set random seeds for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    cfg.ensure_dirs()

    logger.info("Starting FedAvg: dataset=%s model=%s K=%d alpha=%.3f rounds=%d device=%s resume=%s",
                cfg.dataset, cfg.model, cfg.num_clients, cfg.alpha, cfg.rounds, cfg.device, cfg.resume)

    # Load data and distribute to clients
    full_train_dataset, client_indices, test_loader = load_data(cfg)
    logger.info("Data loaded and distributed to clients.")

    # Initialize server
    server = Server(cfg)

    # Determine which clients are honest and which are dishonest
    num_honest_clients = int(cfg.honesty_ratio * cfg.num_clients)
    honest_clients = random.sample(range(cfg.num_clients), num_honest_clients)

    # Collect data sizes for weighting during aggregation
    data_sizes = [len(client_indices[i]) for i in range(cfg.num_clients)]

    # ---- restartability: resume from the last completed round ----
    start_round = 1
    if cfg.resume:
        last = latest_completed_round(cfg.model_dir, 'fedavg')
        if last >= 1:
            server.load_round(last)
            start_round = last + 1
            logger.info("Resuming FedAvg from round %d.", start_round)

    cfg_dict = cfg.to_dict()

    # Federated learning rounds
    for round_num in range(start_round, cfg.rounds + 1):
        logger.info("\n--- Round %d ---", round_num)

        # Distribute global model to clients
        global_model = server.get_global_model()

        # Prepare client states for parallel execution
        client_states = [
            (client_id, cfg_dict, global_model, client_id not in honest_clients)
            for client_id in range(cfg.num_clients)
        ]

        # Clients perform local training in parallel
        with ProcessPoolExecutor() as executor:
            client_models = list(executor.map(train_client, client_states))

        # Server aggregates client models using data sizes
        server.aggregate(client_models, data_sizes)

        # Evaluate global model on test data (also checkpoints the round)
        accuracy = server.evaluate(test_loader, round_num)
        logger.info("Round %d global accuracy: %.2f%%", round_num, accuracy)

    logger.info("Federated Learning completed.")


if __name__ == "__main__":
    main()
