# server_fedavg.py

import torch
import logging
import os
from utils.utils import get_network

class Server:
    def __init__(self, args):
        """
        Initializes the server for federated averaging.

        Args:
            args (Namespace): Parsed arguments containing configurations.
        """
        self.args = args
        self.device = args.device
        self.model = get_network(
            args.model,
            args.channel,
            args.num_classes,
            args.im_size,
            device=self.device
        )
        self.model.to(self.device)
        self.logger = logging.getLogger('FedAvg.Server')
        self.setup_logger()
        self.logger.info("Server initialized with model '%s'.", args.model)

    def setup_logger(self):
        """
        Sets up the logger for the server to log to a separate file.
        """
        log_dir = "/home/t914a431/log/server_logs/"
        os.makedirs(log_dir, exist_ok=True)
        server_log_file = os.path.join(log_dir, 'server_fedavg.log')

        # Prevent adding multiple handlers if already added
        if not self.logger.handlers:
            file_handler = logging.FileHandler(server_log_file)
            console_handler = logging.StreamHandler()

            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)

    def get_global_model_state(self):
        """
        Retrieves the state dictionary of the global model.

        Returns:
            dict: State dictionary of the global model.
        """
        self.logger.info("Server: Retrieving global model state.")
        return self.model.state_dict()

    def aggregate(self, client_models, client_sizes):
        """
        Aggregates client models into the global model using weighted averaging.

        Args:
            client_models (list): List of client model state dictionaries.
            client_sizes (list): List of dataset sizes for each client.
        """
        self.logger.info("Server: Starting aggregation of client models.")
        total_size = sum(client_sizes)
        self.logger.info(f"Server: Total data size across all clients: {total_size}")

        # Initialize the aggregated state dictionary
        aggregated_state_dict = {key: torch.zeros_like(val) for key, val in self.model.state_dict().items()}

        # Weighted aggregation
        for idx, (client_state, client_size) in enumerate(zip(client_models, client_sizes)):
            weight = client_size / total_size
            for key in aggregated_state_dict:
                aggregated_state_dict[key] += client_state[key] * weight
            self.logger.debug(f"Server: Aggregated weights from client {idx} with size {client_size}.")

        # Update the global model
        self.model.load_state_dict(aggregated_state_dict)
        self.logger.info("Server: Global model updated with aggregated client models.")

    def evaluate(self, test_loader, round_num):
        """
        Evaluates the global model on the test dataset.

        Args:
            test_loader (DataLoader): DataLoader for the test dataset.
            round_num (int): Current round number.

        Returns:
            float: Accuracy of the global model on the test dataset.
        """
        self.logger.info(f"Server: Evaluating global model at round {round_num}.")
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        self.logger.info(f"Server: Round {round_num} - Global Model Accuracy: {accuracy:.2f}%")
        return accuracy

    def save_model(self, round_num):
        """
        Saves the global model to disk.

        Args:
            round_num (int): Current round number.
        """
        model_dir = os.path.join('/home/t914a431/models', self.args.dataset, self.args.model, str(self.args.num_clients), str(self.args.honesty_ratio))
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'fedavg_global_model_round_{round_num}.pth')
        try:
            torch.save(self.model.state_dict(), model_path)
            self.logger.info(f"Server: Global model saved to {model_path}.")
        except Exception as e:
            self.logger.error(f"Server: Error saving global model to {model_path} - {e}")
            
